import mlflow
from dotenv import load_dotenv

load_dotenv()
df = mlflow.search_runs(experiment_names=['23L-0616_A2_Ops'])
df = df[df['params.model'].notna() & (df['status'] == 'FINISHED')]
cols = ['tags.mlflow.runName', 'params.model', 'params.img_size', 'params.lr', 'params.augment',
        'params.unfreeze_layers', 'metrics.best_epoch', 'metrics.val_roc_auc', 'metrics.val_recall',
        'metrics.val_real_flagged_as_ai_rate', 'metrics.threshold', 'metrics.test_roc_auc', 'metrics.test_f1']
df = df[[c for c in cols if c in df.columns]].sort_values('metrics.val_roc_auc', ascending=False)
df.to_csv('reports/run_comparison.csv', index=False)
print(df.to_string(index=False))