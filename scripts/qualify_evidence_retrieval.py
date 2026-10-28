#!/usr/bin/env python3
"""Protected evidence retrieval entry point."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT / "src"))
from semantic_retrieval.evidence.cli import main as evidence_main


def main(argv=None):
    return evidence_main(argv,allowed_stages=('cpu_fixture', 'gpu_qualification'),repo_root=Path(__file__).resolve().parents[1])


if __name__ == '__main__':
    raise SystemExit(main())
