import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.config import CLASS_NAMES, FIG_DIR

preds = pd.read_csv(sys.argv[1])            # outputs/<run-name>/test_predictions.csv
wrong = preds[preds.pred != preds.label].copy()
wrong['wrong_conf'] = np.where(wrong.label == 1, 1 - wrong.prob_ai, wrong.prob_ai)
wrong = wrong.sort_values('wrong_conf', ascending=False).head(10)

fig, axes = plt.subplots(2, 5, figsize=(18, 8))
for ax, r in zip(axes.ravel(), wrong.itertuples()):
    with Image.open(r.path) as im:
        ax.imshow(im)
    ax.set_title(f'true {CLASS_NAMES[r.label]} | p(AI)={r.prob_ai:.2f}', fontsize=9)
    ax.axis('off')
fig.tight_layout()
fig.savefig(FIG_DIR / 'misclassified.png', dpi=130)
print(wrong[['source', 'label', 'prob_ai']].to_string(index=False))