import os
import sys

import mlflow
import mlflow.tensorflow
from dotenv import load_dotenv
from mlflow import MlflowClient

from visionguard.preprocess import decode_and_resize, to_clean_jpeg

load_dotenv()
name = os.environ['REGISTERED_MODEL']
uri = f'models:/{name}@champion'
try:
    model = mlflow.tensorflow.load_model(uri)
except Exception:  # noqa: BLE001 - any load failure falls back to the keras flavour
    model = mlflow.keras.load_model(uri)
mv = MlflowClient().get_model_version_by_alias(name, 'champion')
threshold = MlflowClient().get_run(mv.run_id).data.metrics['threshold']

with open(sys.argv[1], 'rb') as f:
    img = decode_and_resize(to_clean_jpeg(f.read()), model.input_shape[1])
p_ai = float(model.predict(img[None, ...], verbose=0)[0, 0])
label = 'AI-generated' if p_ai >= threshold else 'Real'
confidence = p_ai if label == 'AI-generated' else 1 - p_ai
print(f'model {name} v{mv.version} (champion) | threshold {threshold:.3f}')
print(f'{sys.argv[1]} -> {label} (confidence {confidence:.1%}, p_ai={p_ai:.3f})')