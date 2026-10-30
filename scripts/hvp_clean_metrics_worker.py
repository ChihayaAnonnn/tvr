"""CPU-only correction of tie counting on the frozen one-shot score matrix."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from experiment_artifacts import verify_committed_reservation, manifest_sha256
from semantic_retrieval.data.episodes import file_sha256
from semantic_retrieval.evidence.frame_cache import publish_json

REPO = Path(__file__).resolve().parents[1]


def run(root):
    m = json.loads((root/'manifest.json').read_text())
    verify_committed_reservation(REPO, m['registry_path'], m)
    ev = m['config']['evaluation']
    for path, sha in ev['files'].items():
        if file_sha256(path) != sha:
            raise ValueError('changed source/input: '+path)
    parent = Path(ev['parent_root'])
    status = json.loads((parent/'status.json').read_text())
    if status['state'] != 'completed' or status['output_hashes'].get(ev['score_relative']) != ev['files'][ev['score_path']]:
        raise ValueError('matrix is not an archived completed output')
    sys.path.insert(0, ev['hvp_root'])
    from utils.metrics import compute_metrics
    scores = np.load(ev['score_path'], allow_pickle=False)
    if scores.shape != (1000, 1000) or not np.isfinite(scores).all():
        raise ValueError('invalid 1k matrix')
    metrics = {}
    ties = {}
    for direction, matrix in [('t2v', scores), ('v2t', scores.T)]:
        result = compute_metrics(matrix)
        if len(result['cols']) != 1000:
            raise ValueError('must count exactly 1000 queries')
        metrics[direction] = {k: float(v) for k,v in result.items() if k != 'cols'}
        ties[direction] = int(np.sum(np.sum(matrix == np.diag(matrix)[:, None], axis=1) > 1))
    publish_json(root/'provenance/result.json', dict(completed=True, metrics=metrics, ties=ties,
        rank_policy='one rank per query; number of strictly higher scores (optimistic ties)',
        queries=1000, videos=1000, parent_run_id=ev['parent_run_id'], manifest_sha256=manifest_sha256(m)))
    publish_json(root/'metrics/corrected.json', dict(split='test', summary_metrics={d+'_'+k:v for d in metrics for k,v in metrics[d].items()},array_references=[]))
    print(json.dumps(dict(metrics=metrics, ties=ties)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    run(parser.parse_args().run_dir.resolve())
