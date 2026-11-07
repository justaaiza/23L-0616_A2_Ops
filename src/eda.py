import matplotlib
matplotlib.use('Agg')            # draw to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src.config import FIG_DIR, SEED


def sample_grid(df, n=8):
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 5))
    for row, cls in enumerate(['real', 'ai']):
        sample = df[df.cls == cls].sample(n, random_state=SEED)
        for ax, r in zip(axes[row], sample.itertuples()):
            with Image.open(r.path) as im:
                ax.imshow(im.convert('RGB'))
            ax.set_title(f'{cls}  {int(r.width)}x{int(r.height)}', fontsize=8)
            ax.axis('off')
    fig.suptitle('Random samples per class (top: real, bottom: AI-generated)')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'eda_samples.png', dpi=150)


def histograms(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, col, label in zip(axes, ['width', 'height', 'size_kb'], ['width (px)', 'height (px)', 'file size (KB)']):
        for cls in ['real', 'ai']:
            ax.hist(df[df.cls == cls][col], bins=40, alpha=0.55, label=cls)
        ax.set_xlabel(label)
        ax.set_ylabel('number of images')
        ax.legend()
    axes[2].set_xscale('log')       # file sizes span orders of magnitude
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'eda_histograms.png', dpi=150)


if __name__ == '__main__':
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv('reports/audit.csv')
    ok = audit[audit.readable & (audit.cls != 'unknown')]
    sample_grid(ok)
    histograms(ok)
    print('saved figures to', FIG_DIR)