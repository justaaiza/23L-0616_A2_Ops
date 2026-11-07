import datetime
import io

import pytest
from PIL import Image

from visionguard.preprocess import to_clean_jpeg
from web.app import create_app


class FakeClient:
    source = 'test'

    def predict(self, raw, filename):
        to_clean_jpeg(raw)    # same validation as the real model path
        return {'label': 'Real', 'confidence': 0.9, 'prob_ai': 0.1, 'threshold': 0.5,
                'model_name': 'fake', 'model_version': '0', 'latency_ms': 1.0}

    def health(self):
        return {'mode': 'test'}


class FakeStore:
    def __init__(self):
        self.rows = []

    def init(self):
        pass

    def ping(self):
        return True

    def log(self, **r):
        row = {**r, 'id': len(self.rows) + 1, 'created_at': datetime.datetime.now(datetime.UTC)}
        self.rows.append(row)
        return row

    def get(self, prediction_id):
        return next((r for r in self.rows if r['id'] == prediction_id), None)

    def recent(self, limit=100):
        return list(reversed(self.rows))[:limit]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv('MAX_UPLOAD_MB', '1')
    return create_app(client=FakeClient(), store=FakeStore()).test_client()


def jpeg():
    buf = io.BytesIO()
    Image.new('RGB', (64, 64), (0, 128, 255)).save(buf, format='JPEG')
    return buf.getvalue()


def upload(client, data, name):
    return client.post('/predict', data={'image': (io.BytesIO(data), name)},
                       content_type='multipart/form-data', follow_redirects=True)


def test_health(client):
    r = client.get('/health')
    assert r.status_code == 200 and r.json['status'] == 'ok'


def test_valid_upload_is_logged(client):
    r = upload(client, jpeg(), 'cat.jpg')
    assert r.status_code == 200 and b'Real' in r.data
    assert b'cat.jpg' in client.get('/history').data


def test_wrong_type_rejected(client):
    assert upload(client, b'just text', 'notes.txt').status_code == 415


def test_undecodable_image_rejected(client):
    assert upload(client, b'not really a jpeg', 'fake.jpg').status_code == 400


def test_too_large_rejected(client):
    assert upload(client, b'0' * (2 * 1024 * 1024), 'big.jpg').status_code == 413


def test_filename_sanitised(client):
    upload(client, jpeg(), '../../evil.jpg')
    page = client.get('/history').data
    assert b'evil.jpg' in page and b'../' not in page