"""Source code for work of breathing (WOB) experiment project."""
import pathlib

WORKSPACE_DIR = pathlib.Path('/home/workspace/files/preprocessed')
WOB_MEASUREMENTS_PATH = WORKSPACE_DIR / 'measurements' / 'transformed' / 'difficulty_breathing.parquet'
OUTCOMES_PATH = WORKSPACE_DIR / 'outcomes' / 'composite.parquet'
