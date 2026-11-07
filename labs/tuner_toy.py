import keras
import keras_tuner as kt
import pandas as pd

(x, y), _ = keras.datasets.fashion_mnist.load_data()
x = x / 255.0


def build(hp):
    model = keras.Sequential([
        keras.Input((28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(hp.Int('units', 32, 256, step=32), activation='relu'),
        keras.layers.Dense(10, activation='softmax'),
    ])
    model.compile(optimizer=keras.optimizers.Adam(hp.Float('lr', 1e-4, 1e-2, sampling='log')),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


tuner = kt.Hyperband(build, objective='val_accuracy', max_epochs=6, factor=3, seed=42,
                     directory='kt_dir', project_name='fashion_toy', overwrite=True)
tuner.search(x, y, validation_split=0.2, callbacks=[keras.callbacks.EarlyStopping(patience=2)])
tuner.results_summary(5)
print('BEST:', tuner.get_best_hyperparameters(1)[0].values)
rows = [{'trial': t.trial_id, 'units': t.hyperparameters.get('units'), 'lr': t.hyperparameters.get('lr'),
         'val_accuracy': t.score} for t in tuner.oracle.trials.values()]
pd.DataFrame(rows).sort_values('val_accuracy', ascending=False).to_csv('reports/tuner_toy.csv', index=False)