import keras
import keras_tuner as kt
import mlflow
import pandas as pd
from dotenv import load_dotenv

from src.config import IMG_SIZE, SEED
from src.data import make_dataset
from src.models import build_transfer, compile_model, unfreeze_last_block


class VisionGuardHyperModel(kt.HyperModel):
    def build(self, hp):
        model = build_transfer('mobilenetv2', IMG_SIZE, hp.Float('dropout', 0.1, 0.5, step=0.1))
        return compile_model(model, hp.Float('lr', 1e-4, 3e-3, sampling='log'))

    def fit(self, hp, model, *args, **kwargs):
        n = hp.Choice('unfreeze_layers', [0, 10, 20, 40])
        history = model.fit(*args, **kwargs)                 # stage 1: head only
        if n > 0:                                            # stage 2: last n backbone layers
            unfreeze_last_block(model, 'mobilenetv2', n)
            compile_model(model, hp.get('lr') / 100)
            history = model.fit(*args, **kwargs)
        return history


if __name__ == '__main__':
    load_dotenv()
    keras.utils.set_random_seed(SEED)
    train_ds = make_dataset('train', training=True)
    val_ds = make_dataset('val')
    tuner = kt.RandomSearch(VisionGuardHyperModel(), objective='val_loss', max_trials=10, seed=SEED,
                            directory='kt_dir', project_name='23L-0616_A2_Ops_tuner', overwrite=True)
    tuner.search(train_ds, validation_data=val_ds, epochs=6,
                 callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)])
    rows = [{'trial': t.trial_id, **t.hyperparameters.values, 'val_loss': t.score}
            for t in tuner.oracle.trials.values()]
    df = pd.DataFrame(rows).sort_values('val_loss')
    df.to_csv('reports/tuner_visionguard.csv', index=False)
    print(df.to_string(index=False))
    for hp in ['unfreeze_layers', 'dropout']:
        print(df.groupby(hp).val_loss.agg(['mean', 'min', 'count']))
    with mlflow.start_run(run_name='keras-tuner-random-search'):
        mlflow.log_params({f'best_{k}': v for k, v in tuner.get_best_hyperparameters(1)[0].values.items()})
        mlflow.log_artifact('reports/tuner_visionguard.csv', 'tuner')