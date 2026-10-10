#!/usr/bin/env python3
"""Qualified, sealed full-data baseline training worker."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'src'))
from semantic_retrieval.full_data.worker import main

if __name__ == '__main__':
    raise SystemExit(main(ROOT, 'training'))
