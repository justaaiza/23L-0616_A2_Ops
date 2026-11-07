import os

from dotenv import load_dotenv
from flask import Flask, abort, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from web.clients import RemoteAPIError, make_client
from web.db import PostgresStore

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp', 'bmp', 'gif'}


def create_app(client=None, store=None):
    load_dotenv()                     # local runs; in Docker the env comes from Compose
    app = Flask(__name__)
    max_mb = int(os.getenv('MAX_UPLOAD_MB', '5'))
    app.config.update(SECRET_KEY=os.getenv('SECRET_KEY', 'dev-only'), MAX_CONTENT_LENGTH=max_mb * 1024 * 1024)
    client = client or make_client()
    store = store or PostgresStore(os.environ['DATABASE_URL'])
    store.init()

    def fail(message, status):
        return render_template('error.html', message=message, status=status), status

    @app.get('/')
    def index():
        return render_template('index.html', max_mb=max_mb, allowed=sorted(ALLOWED_EXTENSIONS))

    @app.post('/predict')
    def predict():
        upload = request.files.get('image')
        if upload is None or upload.filename == '':
            return fail('Please choose an image to upload.', 400)
        filename = secure_filename(upload.filename) or 'upload'
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        if ext not in ALLOWED_EXTENSIONS:
            return fail(f'".{ext or "?"}" files are not accepted. Upload one of: {", ".join(sorted(ALLOWED_EXTENSIONS))}.', 415)
        raw = upload.read()           # bytes in memory only; never saved to disk
        try:
            result = client.predict(raw, filename)
        except ValueError:
            return fail('This file has an image extension but could not be decoded as an image. '
                        'It may be corrupted or not really an image.', 400)
        except RemoteAPIError:
            app.logger.exception('model API call failed')
            return fail('The model service is not reachable right now. Please try again in a minute.', 503)
        row = store.log(filename=filename, source=client.source, **result)
        return redirect(url_for('result', prediction_id=row['id']))

    @app.get('/result/<int:prediction_id>')
    def result(prediction_id):
        row = store.get(prediction_id) or abort(404)
        return render_template('result.html', r=row)

    @app.get('/history')
    def history():
        return render_template('history.html', rows=store.recent(100))

    @app.get('/health')
    def health():
        db_ok = store.ping()
        body = {'status': 'ok' if db_ok else 'degraded', 'database': 'ok' if db_ok else 'unreachable',
                **client.health(), 'threshold': os.getenv('THRESHOLD'), 'max_upload_mb': max_mb}
        return jsonify(body), (200 if db_ok else 503)

    @app.errorhandler(413)
    def too_large(_):
        return fail(f'That file is larger than the {max_mb} MB limit. Please upload a smaller image.', 413)

    @app.errorhandler(404)
    def not_found(_):
        return fail('That page does not exist.', 404)

    @app.errorhandler(405)
    def bad_method(_):
        return fail('That action is not allowed here.', 405)

    @app.errorhandler(500)
    def server_error(_):
        return fail('Something went wrong on our side. The error has been logged.', 500)

    return app