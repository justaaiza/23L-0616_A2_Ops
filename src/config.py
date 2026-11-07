from pathlib import Path

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32

RAW_DIR = Path('data/raw')
CLEAN_DIR = Path('data/clean')
SPLIT_DIR = Path('data/splits')
FIG_DIR = Path('reports/figures')

LABELS = {'real': 0, 'ai': 1}        # sigmoid output = P(AI-generated)
CLASS_NAMES = ['Real', 'AI-generated']  # index = label