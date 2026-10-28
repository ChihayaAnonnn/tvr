"""Freeze an independent AQS run using authenticated MSRVTT inputs."""
import argparse
from dataclasses import asdict
from pathlib import Path
import subprocess

from semantic_retrieval.aqs.launch import validate_recipe
from semantic_retrieval.data.episodes import file_sha256
from semantic_retrieval.evidence.frame_cache import publish_json
from semantic_retrieval.full_data.campaign import scientific_inputs, PRETRAINED, FORMAL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--negatives', type=int, required=True)
    parser.add_argument('--weight', type=float, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    recipe = dict(epochs=5, global_batch=64, learning_rate=5e-5, seed=713, world_size=2,
                  negatives=args.negatives, weight=args.weight, validation_only=True)
    validate_recipe(recipe)
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD'], cwd=repo, check=True)
    inputs, certificate = scientific_inputs(repo)
    weights_sha = file_sha256(PRETRAINED)
    if weights_sha != '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af':
        raise ValueError('unexpected pretrained CLIP bytes')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    names = subprocess.check_output(['git', 'ls-files', 'src', 'scripts', 'experiment_artifacts.py'], cwd=repo, text=True).splitlines()
    files = {n: file_sha256(repo/n) for n in names if n.endswith('.py')}
    root = args.run_dir.resolve(); root.mkdir(parents=True, exist_ok=False)
    publish_json(root/'manifest.json', dict(schema='uatvr-aqs-training-v1', source_sha=source,
        source_files=files, recipe=recipe, clip=asdict(FORMAL), dataset_splits=inputs,
        input_checkpoints=[dict(path=str(PRETRAINED), sha256=weights_sha)],
        source_report=dict(path=str(certificate), sha256=file_sha256(certificate)),
        test_authorized=False, laya_enabled=False))
    print(root)


if __name__ == '__main__':
    main()
