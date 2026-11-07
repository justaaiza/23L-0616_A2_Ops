"""Walk data/raw and build one audit row per file."""
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image

from src.config import RAW_DIR

OUT = Path('reports/audit.csv')


def class_of(path: Path) -> str:
    top = path.relative_to(RAW_DIR).parts[0].lower()
    if 'real' in top:
        return 'real'
    if 'ai' in top:
        return 'ai'
    return 'unknown'


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def audit_file(path: Path) -> dict:
    row = {'path': str(path), 'cls': class_of(path), 'size_bytes': path.stat().st_size,
           'sha256': sha256_of(path), 'width': None, 'height': None, 'mode': None,
           'format': None, 'readable': True, 'error': ''}
    try:
        with Image.open(path) as im:
            im.verify()                      # checks file integrity
        with Image.open(path) as im:         # verify() invalidates the object, so reopen
            im.load()                        # forces a full decode
            row.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
    except Exception as exc:                 # truncated, not an image, decompression bomb...
        row.update(readable=False, error=f'{type(exc).__name__}: {exc}')
    return row


def main():
    files = [p for p in sorted(RAW_DIR.rglob('*')) if p.is_file() and not p.name.startswith('.')]
    df = pd.DataFrame([audit_file(p) for p in files])
    df['size_kb'] = df.size_bytes / 1024
    df['is_extra_copy'] = df.duplicated('sha256', keep='first')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    ok = df[df.readable]
    print('== files per class ==')
    print(df.groupby('cls').agg(files=('path', 'count'), readable=('readable', 'sum')))
    print('== width / height / size (KB), readable images ==')
    print(ok.groupby('cls')[['width', 'height', 'size_kb']].agg(['min', 'median', 'max']).round(1))
    print('corrupt / unreadable files:', int((~df.readable).sum()))
    print('exact duplicates (extra copies):', int(df.is_extra_copy.sum()))
    both = df.groupby('sha256').cls.nunique()
    print('identical files found in BOTH classes:', int((both > 1).sum()))
    print('== formats ==')
    print(df.groupby(['cls', 'format'], dropna=False).size())
    print('== colour modes ==')
    print(df.groupby(['cls', 'mode'], dropna=False).size())
    print(df[~df.readable][['path', 'error']].to_string())


if __name__ == '__main__':
    main()