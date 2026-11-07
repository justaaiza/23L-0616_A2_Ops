import argparse

import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
from dotenv import load_dotenv

p = argparse.ArgumentParser()
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--remote', action='store_true', help='log to DagsHub instead of ./mlruns')
args = p.parse_args()

if args.remote:
    load_dotenv()                            # reads MLFLOW_TRACKING_URI/USERNAME/PASSWORD from .env
else:
    mlflow.set_tracking_uri('file:./mlruns')
mlflow.set_experiment('23L-0616_A2_Ops_toy')

(x, y), (xt, yt) = keras.datasets.mnist.load_data()
x, xt = x[:10000] / 255.0, xt[:2000] / 255.0
y, yt = y[:10000], yt[:2000]
model = keras.Sequential([keras.Input((28, 28)), keras.layers.Flatten(),
                          keras.layers.Dense(64, activation='relu'), keras.layers.Dense(10, activation='softmax')])
model.compile(keras.optimizers.Adam(args.lr), 'sparse_categorical_crossentropy', metrics=['accuracy'])


class EpochLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in logs.items()}, step=epoch + 1)


with mlflow.start_run(run_name=f'toy-lr{args.lr}'):
    mlflow.set_tag('purpose', 'toy lab 3.1')
    mlflow.log_params({'lr': args.lr, 'epochs': 5, 'hidden_units': 64})
    hist = model.fit(x, y, validation_data=(xt, yt), epochs=5, callbacks=[EpochLogger()], verbose=2)
    fig, ax = plt.subplots()
    ax.plot(hist.history['loss'], label='train')
    ax.plot(hist.history['val_loss'], label='val')
    ax.set_xlabel('epoch'); ax.set_ylabel('loss'); ax.legend()
    mlflow.log_figure(fig, 'loss_curve.png')
    try:
        mlflow.tensorflow.log_model(model, artifact_path='model')
    except Exception:
        mlflow.keras.log_model(model, artifact_path='model')