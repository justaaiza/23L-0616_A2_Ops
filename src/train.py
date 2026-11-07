"""Train a VisionGuard model and track everything in MLflow."""
import argparse
import os
import subprocess
from pathlib import Path

import keras
import mlflow
import mlflow.tensorflow
import pandas as pd
import tensorflow as tf
import yaml
from dotenv import load_dotenv

from src import evaluate as ev
from src.config import BATCH_SIZE, SEED, SPLIT_DIR
from src.data import make_dataset
from src.models import build_baseline, build_transfer, compile_model, trainable_layer_names, unfreeze_last_block


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='mobilenetv2', choices=['baseline', 'mobilenetv2', 'efficientnetb0'])
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--ft-lr', type=float, default=1e-5)
    p.add_argument('--head-epochs', type=int, default=25)
    p.add_argument('--ft-epochs', type=int, default=15)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--unfreeze-layers', type=int, default=None, help='omit = last block; 0 = no fine-tuning')
    p.add_argument('--no-augment', action='store_true')
    p.add_argument('--logging', choices=['manual', 'autolog'], default='manual')
    p.add_argument('--max-fpr', type=float, default=0.05)
    p.add_argument('--run-name', default=None)
    return p.parse_args()


def git_info():
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True).stdout.strip())
    return commit or 'unknown', dirty


def dvc_version():
    try:
        return yaml.safe_load(Path('data/clean.dvc').read_text())['outs'][0]['md5']
    except Exception:
        return 'unknown'


class MlflowEpochLogger(keras.callbacks.Callback):
    """Manual logging: every epoch's train/val metrics, with a continuous step across stages."""
    def __init__(self, offset=0):
        super().__init__()
        self.offset = offset

    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=self.offset + epoch + 1)


def log_keras_model(model):
    try:
        mlflow.tensorflow.log_model(model, artifact_path='model')
    except Exception as exc:  # some MLflow/Keras 3 combinations need the keras flavour
        print('tensorflow flavour failed, using keras flavour:', exc)
        mlflow.keras.log_model(model, artifact_path='model')


def main():
    args = parse_args()
    load_dotenv()
    keras.utils.set_random_seed(SEED)              # seeds Python, NumPy and TensorFlow together
    tf.config.experimental.enable_op_determinism()  # same seed -> same result (slightly slower)
    mlflow.set_experiment(os.getenv('MLFLOW_EXPERIMENT', '23L-0616_A2_Ops'))
    if args.logging == 'autolog':
        mlflow.tensorflow.autolog(log_models=True, log_datasets=False)

    augment = not args.no_augment
    name = args.run_name or f"{args.model}-{args.img_size}-lr{args.lr}-{'aug' if augment else 'noaug'}-{args.logging}"
    train_ds = make_dataset('train', args.img_size, training=True, augment=augment)
    val_ds = make_dataset('val', args.img_size)
    test_ds = make_dataset('test', args.img_size)
    y_train = pd.read_csv(SPLIT_DIR / 'train.csv').label.to_numpy()
    class_weight = {c: len(y_train) / (2 * (y_train == c).sum()) for c in (0, 1)}

    commit, dirty = git_info()
    out = Path('outputs') / name
    out.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({'git_commit': commit, 'git_dirty': str(dirty), 'dvc_data_md5': dvc_version(),
                         'seed': str(SEED), 'logging_mode': args.logging, 'model_family': args.model,
                         'roll_number': '23L-0616'})
        mlflow.log_params({'model': args.model, 'img_size': args.img_size, 'lr': args.lr, 'ft_lr': args.ft_lr,
                           'head_epochs': args.head_epochs, 'ft_epochs': args.ft_epochs, 'dropout': args.dropout,
                           'unfreeze_layers': 'last_block' if args.unfreeze_layers is None else args.unfreeze_layers,
                           'augment': augment, 'batch': BATCH_SIZE, 'max_fpr': args.max_fpr,
                           'class_weight_real': round(class_weight[0], 3), 'class_weight_ai': round(class_weight[1], 3)})

        if args.model == 'baseline':
            model = build_baseline(args.img_size)
        else:
            model = build_transfer(args.model, args.img_size, args.dropout)
        compile_model(model, args.lr)

        def fit_stage(stage, epochs, offset):
            ckpt = str(out / f'best_{stage}.weights.h5')
            cbs = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                   keras.callbacks.ModelCheckpoint(ckpt, monitor='val_loss', save_best_only=True, save_weights_only=True),
                   keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-7)]
            if args.logging == 'manual':
                cbs.append(MlflowEpochLogger(offset))
            mlflow.log_text('\n'.join(trainable_layer_names(model)), f'trainable_layers_{stage}.txt')
            kwargs = dict(validation_data=val_ds, epochs=epochs, callbacks=cbs, class_weight=class_weight, verbose=2)
            if args.logging == 'autolog':
                with mlflow.start_run(run_name=stage, nested=True):  # each fit gets its own child run
                    hist = model.fit(train_ds, **kwargs)
            else:
                hist = model.fit(train_ds, **kwargs)
            model.load_weights(ckpt)  # continue from the best val_loss epoch
            return hist.history

        histories = [fit_stage('stage1', args.head_epochs, 0)]
        if args.model != 'baseline' and args.unfreeze_layers != 0:
            unfreeze_last_block(model, args.model, args.unfreeze_layers)
            compile_model(model, args.ft_lr)
            histories.append(fit_stage('stage2', args.ft_epochs, len(histories[0]['loss'])))

        best_epoch = ev.plot_history(histories, out / 'curves.png', name)
        mlflow.log_metric('best_epoch', best_epoch)
        mlflow.log_metric('best_val_loss', float(min(sum([h['val_loss'] for h in histories], []))))

        # threshold chosen on VALIDATION only
        y_val, p_val = ev.predict(model, val_ds)
        t = ev.choose_threshold(y_val, p_val, args.max_fpr)
        mlflow.log_metric('threshold', t)
        mlflow.log_metrics({f'val_{k}': float(v) for k, v in ev.metrics(y_val, p_val, t).items()})

        # test set: reported, never used to choose anything
        y_test, p_test = ev.predict(model, test_ds)
        mlflow.log_metrics({f'test_{k}': float(v) for k, v in ev.metrics(y_test, p_test, t).items()})
        mlflow.log_metrics({f'test05_{k}': float(v) for k, v in ev.metrics(y_test, p_test, 0.5).items()})
        ev.plot_confusion(y_test, p_test, t, out / 'confusion_matrix.png')
        ev.plot_roc(y_test, p_test, t, out / 'roc_curve.png')

        preds = pd.read_csv(SPLIT_DIR / 'test.csv')
        preds['prob_ai'] = p_test
        preds['pred'] = (p_test >= t).astype(int)
        preds.to_csv(out / 'test_predictions.csv', index=False)

        model.save(out / 'model.keras')
        for f in ['curves.png', 'confusion_matrix.png', 'roc_curve.png', 'test_predictions.csv']:
            mlflow.log_artifact(str(out / f), 'evaluation')
        mlflow.log_artifact(str(out / 'model.keras'), 'model_file')
        mlflow.log_artifacts(str(SPLIT_DIR), 'splits')
        subprocess.run('pip freeze > requirements-lock.txt', shell=True)
        mlflow.log_artifact('requirements-lock.txt', 'environment')
        log_keras_model(model)
        print(f'run {run.info.run_id}  best epoch {best_epoch}  threshold {t:.3f}  outputs in {out}')


if __name__ == '__main__':
    main()