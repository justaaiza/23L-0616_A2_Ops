import keras
import pandas as pd
from keras import layers

LAST_BLOCK_START = {'mobilenetv2': 'block_16_expand', 'efficientnetb0': 'block7a_expand_conv'}


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc'),
                 keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')],
    )
    return model


def build_baseline(img_size=224, dropout=0.4):
    inputs = keras.Input((img_size, img_size, 3), name='image')
    x = layers.Rescaling(1 / 255.0, name='preprocess')(inputs)
    for i, filters in enumerate([32, 64, 128, 256], start=1):
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False, name=f'conv{i}')(x)
        x = layers.BatchNormalization(name=f'bn{i}')(x)
        x = layers.ReLU(name=f'relu{i}')(x)
        x = layers.MaxPooling2D(name=f'pool{i}')(x)
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dropout(dropout, name='head_dropout')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='pred')(x)
    return keras.Model(inputs, outputs, name='baseline_cnn')


def build_transfer(backbone='mobilenetv2', img_size=224, dropout=0.3):
    inputs = keras.Input((img_size, img_size, 3), name='image')
    shape = (img_size, img_size, 3)
    if backbone == 'mobilenetv2':
        # identical to mobilenet_v2.preprocess_input: [0,255] -> [-1,1]
        x = layers.Rescaling(1 / 127.5, offset=-1.0, name='preprocess')(inputs)
        base = keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=shape)
    elif backbone == 'efficientnetb0':
        # EfficientNet normalises internally and expects raw [0,255] pixels
        x = layers.Identity(name='preprocess')(inputs)
        base = keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_shape=shape)
    else:
        raise ValueError(backbone)
    base.trainable = False
    x = base(x, training=False)   # BatchNorm stays in inference mode, even after unfreezing
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dropout(dropout, name='head_dropout')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='pred')(x)
    return keras.Model(inputs, outputs, name=f'{backbone}_transfer')


def get_backbone(model):
    return next(l for l in model.layers if isinstance(l, keras.Model))


def unfreeze_last_block(model, backbone, n_layers=None):
    """n_layers=None: unfreeze from the start of the last block. Otherwise the last n_layers."""
    base = get_backbone(model)
    base.trainable = True
    names = [l.name for l in base.layers]
    start = names.index(LAST_BLOCK_START[backbone]) if n_layers is None else len(names) - n_layers
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= start and not isinstance(layer, layers.BatchNormalization)


def trainable_layer_names(model):
    out = [l.name for l in model.layers if l.trainable and l.weights and not isinstance(l, keras.Model)]
    for sub in (l for l in model.layers if isinstance(l, keras.Model)):
        out += [f'{sub.name}/{l.name}' for l in sub.layers if l.trainable and l.weights]
    return out


def layer_table(model):
    return pd.DataFrame([{'layer': l.name, 'type': type(l).__name__,
                          'output_shape': str(l.output.shape), 'params': l.count_params()}
                         for l in model.layers])


if __name__ == '__main__':
    for m in [build_baseline(), build_transfer('mobilenetv2')]:
        m.summary()
        layer_table(m).to_csv(f'reports/layers_{m.name}.csv', index=False)
        print(layer_table(m).to_string(index=False))