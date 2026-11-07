import os

import requests


class RemoteAPIError(Exception):
    """The model API could not be reached or answered with an error."""


class LocalClient:
    source = 'local'

    def __init__(self):
        from visionguard.inference import Predictor
        self.predictor = Predictor.from_env()

    def predict(self, raw, filename):
        return self.predictor.predict(raw)

    def health(self):
        return {'mode': 'local', 'model_version': self.predictor.model_version}


class RemoteClient:
    source = 'remote-api'

    def __init__(self, url, key, timeout=30):
        self.url, self.key, self.timeout = url.rstrip('/'), key, timeout

    def predict(self, raw, filename):
        try:
            r = requests.post(f'{self.url}/predict', files={'file': (filename, raw)},
                              headers={'X-API-Key': self.key}, timeout=self.timeout)
        except requests.RequestException as exc:
            raise RemoteAPIError(f'model API unreachable: {exc}') from exc
        if r.status_code in (400, 413, 415):
            raise ValueError(r.json().get('error', 'invalid image'))
        if r.status_code != 200:
            raise RemoteAPIError(f'model API returned {r.status_code}: {r.text[:200]}')
        return r.json()

    def health(self):
        try:
            r = requests.get(f'{self.url}/health', timeout=5)
            return {'mode': 'remote', 'api_url': self.url, 'api_status': r.status_code}
        except requests.RequestException as exc:
            return {'mode': 'remote', 'api_url': self.url, 'api_status': f'unreachable: {exc}'}


def make_client():
    if os.getenv('INFERENCE_MODE', 'local') == 'remote':
        return RemoteClient(os.environ['API_URL'], os.environ['API_KEY'])
    return LocalClient()