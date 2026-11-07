import platform
import sys

import keras
import mlflow
import numpy
import tensorflow as tf

print('Python     :', sys.version.split()[0], platform.platform())
print('TensorFlow :', tf.__version__)
print('Keras      :', keras.__version__)
print('NumPy      :', numpy.__version__)
print('MLflow     :', mlflow.__version__)
print('GPUs       :', tf.config.list_physical_devices('GPU') or 'none (CPU only)')
print('Built with CUDA:', tf.test.is_built_with_cuda())