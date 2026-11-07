import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras import layers

from src.config import BATCH_SIZE, CLASS_NAMES, FIG_DIR, IMG_SIZE, SEED, SPLIT_DIR
from visionguard.preprocess import decode_and_resize

AUTOTUNE = tf.data.AUTOTUNE


def make_augmenter():
    # mild on purpose: strong blur/noise/JPEG would erase the generator artefacts we want to detect
    return keras.Sequential([
        layers.RandomFlip('horizontal', seed=SEED),
        layers.RandomRotation(0.05, seed=SEED),
        layers.RandomZoom(0.10, seed=SEED),
        layers.RandomContrast(0.10, seed=SEED),
    ], name='augment')


def load_split(name):
    df = pd.read_csv(SPLIT_DIR / f'{name}.csv')
    return df.path.tolist(), df.label.astype('float32').tolist()


def make_dataset(name, img_size=IMG_SIZE, training=False, augment=True, batch_size=BATCH_SIZE):
    paths, labels = load_split(name)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))           # 1. file list
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)  # 2. shuffle
    ds = ds.map(lambda p, y: (decode_and_resize(tf.io.read_file(p), img_size), y),
                num_parallel_calls=AUTOTUNE)                            # 3. read + decode + resize
    ds = ds.batch(batch_size)                                         # 4. batch
    if training and augment:
        aug = make_augmenter()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)  # 5. augment
    return ds.prefetch(AUTOTUNE)                                       # 6. prefetch


if __name__ == '__main__':
    images, labels = next(iter(make_dataset('train', training=True)))
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    for ax, img, y in zip(axes.ravel(), images, labels):
        ax.imshow(img.numpy().astype('uint8'))
        ax.set_title(CLASS_NAMES[int(y)], fontsize=8)
        ax.axis('off')
    fig.suptitle('One augmented training batch')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'augmented_batch.png', dpi=130)
    print('saved', FIG_DIR / 'augmented_batch.png')