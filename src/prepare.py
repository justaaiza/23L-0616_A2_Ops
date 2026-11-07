"""Clean images into data/clean and write stratified split lists."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import CLEAN_DIR, LABELS, SEED, SPLIT_DIR
from visionguard.preprocess import to_clean_jpeg


def main():
    audit = pd.read_csv('reports/audit.csv')
    ok = audit[audit.readable & (audit.cls != 'unknown')]
    # same bytes in both classes = contradictory label -> drop every copy
    n_cls = ok.groupby('sha256').cls.nunique()
    ok = ok[~ok.sha256.isin(n_cls[n_cls > 1].index)]
    # keep one copy of each duplicate BEFORE splitting, so no image is in two splits
    ok = ok.drop_duplicates('sha256', keep='first')

    rows, skipped = [], 0
    for r in ok.itertuples():
        try:
            jpeg = to_clean_jpeg(Path(r.path).read_bytes())
        except ValueError as exc:
            skipped += 1
            print('skip', r.path, exc)
            continue
        out = CLEAN_DIR / r.cls / f'{r.sha256[:16]}.jpg'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(jpeg)
        rows.append({'path': str(out), 'label': LABELS[r.cls], 'cls': r.cls, 'source': r.path})

    df = pd.DataFrame(rows)
    train, tmp = train_test_split(df, test_size=0.30, stratify=df.label, random_state=SEED)
    val, test = train_test_split(tmp, test_size=0.50, stratify=tmp.label, random_state=SEED)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in [('train', train), ('val', val), ('test', test)]:
        part.to_csv(SPLIT_DIR / f'{name}.csv', index=False)
        print(f'{name:5s} {len(part):4d}', part.cls.value_counts().to_dict())
    print('skipped (too small / undecodable):', skipped)


if __name__ == '__main__':
    main()