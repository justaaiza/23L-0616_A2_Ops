import sys

import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from src.config import CLASS_NAMES, FIG_DIR
from src.models import get_backbone
from visionguard.preprocess import decode_and_resize


def gradcam(model, img, for_ai=True):
    backbone = get_backbone(model)
    head = [model.get_layer(n) for n in ('gap', 'head_dropout', 'pred')]
    x = model.get_layer('preprocess')(img[None, ...])
    with tf.GradientTape() as tape:
        fmap = backbone(x, training=False)    # last conv feature map, e.g. 7x7x1280
        tape.watch(fmap)
        y = fmap
        for layer in head:
            y = layer(y, training=False)
        score = y[:, 0] if for_ai else 1 - y[:, 0]
    grads = tape.gradient(score, fmap)
    weights = tf.reduce_mean(grads, axis=(1, 2))                 # importance of each channel
    cam = tf.nn.relu(tf.reduce_sum(fmap * weights[:, None, None, :], axis=-1))[0]
    return (cam / (tf.reduce_max(cam) + 1e-8)).numpy()


model = keras.models.load_model(sys.argv[1])        # outputs/<run>/model.keras
preds = pd.read_csv(sys.argv[2])                    # outputs/<run>/test_predictions.csv
size = model.input_shape[1]
picks = pd.concat([preds[preds.pred == preds.label].head(4), preds[preds.pred != preds.label].head(4)])
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, r in zip(axes.ravel(), picks.itertuples()):
    img = decode_and_resize(tf.io.read_file(r.path), size)
    cam = gradcam(model, img, for_ai=bool(r.pred))
    ax.imshow(img.numpy().astype('uint8'))
    ax.imshow(tf.image.resize(cam[..., None], (size, size)).numpy()[..., 0], cmap='jet', alpha=0.4)
    status = 'correct' if r.pred == r.label else 'WRONG'
    ax.set_title(f'{status}: true {CLASS_NAMES[r.label]}, p(AI)={r.prob_ai:.2f}', fontsize=9)
    ax.axis('off')
fig.tight_layout()
fig.savefig(FIG_DIR / 'gradcam.png', dpi=130)