"""VisionGuard model API: loads the champion once, returns JSON verdicts."""
import hmac
import logging
import os

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

from visionguard.inference import Predictor

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp', 'bmp', 'gif'}


def create_app(predictor=None):
    app = Flask(__name__)
    app.logger.setLevel(logging.INFO)
    max_mb = int(os.getenv('MAX_UPLOAD_MB', '5'))
    app.config['MAX_CONTENT_LENGTH'] = max_mb * 1024 * 1024
    api_key = os.environ['API_KEY']           # refuse to start without a key
    predictor = predictor or Predictor.from_env()

    def error(message, status):
        return jsonify({'error': message}), status

    @app.get('/health')
    def health():
        return jsonify({'status': 'ok', 'model_name': predictor.model_name,
                        'model_version': predictor.model_version, 'threshold': predictor.threshold})

    @app.post('/predict')
    def predict():
        if not hmac.compare_digest(request.headers.get('X-API-Key', ''), api_key):
            return error('missing or invalid API key', 401)
        upload = request.files.get('file')
        if upload is None or upload.filename == '':
            return error("send the image as multipart form field 'file'", 400)
        name = secure_filename(upload.filename) or 'upload'
        ext = name.rsplit('.', 1)[-1].lower() if '.' in name else ''
        if ext not in ALLOWED_EXTENSIONS:
            return error(f'file type .{ext} not allowed', 415)
        try:
            result = predictor.predict(upload.read())
        except ValueError as exc:
            return error(f'could not decode image: {exc}', 400)
        app.logger.info('predict file=%s label=%s p_ai=%.3f %.0fms from=%s', name, result['label'],
                        result['prob_ai'], result['latency_ms'], request.remote_addr)
        return jsonify(result)

    @app.errorhandler(413)
    def too_large(_):
        return error(f'file larger than {max_mb} MB', 413)

    @app.errorhandler(404)
    def not_found(_):
        return error('not found', 404)

    @app.errorhandler(405)
    def bad_method(_):
        return error('method not allowed', 405)

    return app