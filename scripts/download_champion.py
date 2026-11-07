import os
import shutil
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient

load_dotenv()
name = os.environ['REGISTERED_MODEL']
client = MlflowClient()
mv = client.get_model_version_by_alias(name, 'champion')
src = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path='model_file/model.keras', dst_path='models/_download')
Path('models').mkdir(exist_ok=True)
shutil.copy(src, 'models/champion.keras')
print('saved models/champion.keras')
print(f'MODEL_VERSION={mv.version}')
print(f"THRESHOLD={client.get_run(mv.run_id).data.metrics['threshold']:.4f}")