"""Loads the champion model ONCE and turns raw image bytes into a verdict."""
import os
import sys
import time

import keras
import tensorflow as tf

from visionguard.preprocess import decode_and_resize, to_clean_jpeg


class Predictor:
    def __init__(self, model_path, threshold=0.5, model_name='visionguard', model_version='unknown'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model file not found: {model_path}')
        t0 = time.perf_counter()
        self.model = keras.models.load_model(model_path)
        self.img_size = self.model.input_shape[1]
        self.threshold = float(threshold)
        self.model_name, self.model_version = model_name, str(model_version)
        self.model(tf.zeros((1, self.img_size, self.img_size, 3)), training=False)  # warm-up call
        print(f'[inference] loaded {model_path} (v{self.model_version}) in {time.perf_counter() - t0:.1f}s', flush=True)

    @classmethod
    def from_env(cls):
        return cls(os.environ['MODEL_PATH'], os.getenv('THRESHOLD', '0.5'),
                   os.getenv('MODEL_NAME', 'visionguard'), os.getenv('MODEL_VERSION', 'unknown'))

    def predict(self, raw: bytes) -> dict:
        t0 = time.perf_counter()
        x = decode_and_resize(to_clean_jpeg(raw), self.img_size)   # SAME code as training; ValueError if not an image
        p_ai = float(self.model(x[None, ...], training=False).numpy()[0, 0])
        is_ai = p_ai >= self.threshold
        return {'label': 'AI-generated' if is_ai else 'Real',
                'confidence': p_ai if is_ai else 1.0 - p_ai,
                'prob_ai': p_ai, 'threshold': self.threshold,
                'model_name': self.model_name, 'model_version': self.model_version,
                'latency_ms': (time.perf_counter() - t0) * 1000}


if __name__ == '__main__':          # python -m visionguard.inference img1 img2 ...
    predictor = Predictor.from_env()
    for path in sys.argv[1:]:
        with open(path, 'rb') as f:
            print(path, '->', predictor.predict(f.read()))