#!/usr/bin/env python3
"""Execute an already committed and reserved full-data phase."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'src'))
from semantic_retrieval.full_data.controller import run_phase

if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--phase', type=Path, required=True)
    parser.add_argument('--artifact-root', type=Path, default=ROOT/'artifacts')
    args = parser.parse_args()
    raise SystemExit(run_phase(ROOT, args.phase, args.artifact_root))
