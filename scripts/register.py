"""Usage: python scripts/register.py <run_id_v1> <run_id_v2> ...  (last one becomes champion)"""
import os
import sys

import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient

load_dotenv()
name = os.environ['REGISTERED_MODEL']          # 23L-0616_A2_Ops_VisionGuard
versions = [mlflow.register_model(f'runs:/{rid}/model', name) for rid in sys.argv[1:]]
for v in versions:
    print('registered', name, 'version', v.version, 'from run', v.run_id)
MlflowClient().set_registered_model_alias(name, 'champion', versions[-1].version)
print('alias champion ->', versions[-1].version)