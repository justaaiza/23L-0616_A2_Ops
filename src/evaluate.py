import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from src.config import CLASS_NAMES


def predict(model, ds):
    y_true = np.concatenate([y.numpy() for _, y in ds]).astype(int)
    y_prob = model.predict(ds, verbose=0).ravel()
    return y_true, y_prob


def choose_threshold(y_true, y_prob, max_fpr=0.05):
    """Highest recall on AI images while flagging at most max_fpr of REAL images as AI."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    ok = np.where(fpr <= max_fpr)[0]
    best = ok[np.argmax(tpr[ok])]
    if tpr[best] == 0:  # constraint impossible -> fall back to best F1
        cands = np.linspace(0.05, 0.95, 91)
        return float(cands[np.argmax([f1_score(y_true, y_prob >= c) for c in cands])])
    return float(min(thr[best], 1.0))


def metrics(y_true, y_prob, t):
    y_pred = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'real_flagged_as_ai_rate': fp / (fp + tn), 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}


def plot_confusion(y_true, y_prob, t, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, (y_prob >= t).astype(int),
                                            display_labels=CLASS_NAMES, ax=ax, colorbar=False)
    ax.set_title(f'Test confusion matrix (threshold {t:.2f})')
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_roc(y_true, y_prob, t, path):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    i = np.argmin(np.abs(thr - t))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_true, y_prob):.3f}')
    ax.plot([0, 1], [0, 1], '--', color='grey')
    ax.scatter(fpr[i], tpr[i], color='red', zorder=3, label=f'chosen threshold {t:.2f}')
    ax.set_xlabel('False positive rate (real called AI)')
    ax.set_ylabel('True positive rate (AI caught)')
    ax.set_title('Test ROC curve'); ax.legend(loc='lower right')
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_history(histories, path, title):
    loss = sum([h['loss'] for h in histories], [])
    val_loss = sum([h['val_loss'] for h in histories], [])
    acc = sum([h['accuracy'] for h in histories], [])
    val_acc = sum([h['val_accuracy'] for h in histories], [])
    epochs = np.arange(1, len(loss) + 1)
    best = int(np.argmin(val_loss)) + 1
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (tr, va, name) in zip(axes, [(loss, val_loss, 'loss'), (acc, val_acc, 'accuracy')]):
        ax.plot(epochs, tr, label=f'train {name}')
        ax.plot(epochs, va, label=f'val {name}')
        ax.axvline(best, color='red', ls='--', label=f'best epoch {best}')
        if len(histories) > 1:
            ax.axvline(len(histories[0]['loss']) + 0.5, color='grey', ls=':', label='fine-tuning starts')
        ax.set_xlabel('epoch'); ax.set_ylabel(name); ax.legend()
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return best