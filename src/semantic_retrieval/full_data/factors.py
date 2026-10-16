"""Predeclared single-factor policies; randomness is independent of worker scheduling."""
import hashlib
import json
import numpy as np

ARMS = ('control', 'temporal_offset', 'caption_shuffle')


def offset_frame_indices(frame_count, fps, max_frames, *, seed, epoch, query_id):
    from .loader import official_frame_indices
    base = official_frame_indices(frame_count, fps, max_frames)
    if (type(seed) is not int or seed < 0 or type(epoch) is not int or epoch < 0
            or not isinstance(query_id, str) or not query_id):
        raise ValueError('invalid stateless temporal sampling identity')
    raw = json.dumps([seed, epoch, query_id], separators=(',', ':')).encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big'))
    # Same 29 possible temporal phases as upstream randint(0,29), adapted to
    # native FPS. Cap at the final frame to preserve baseline observation count.
    offset = int(int(rng.integers(0,29))*int(fps)/30)
    return [min(i+offset, frame_count-1) for i in base]


def caption_batches(identities, *, global_batch, world_size, seed, epoch):
    from .training import unique_video_batches
    # Reuse its strict cohort checks and identical retained batch cardinalities.
    sizes = [len(b) for b in unique_video_batches(identities, global_batch=global_batch,
                                               world_size=world_size, seed=seed, epoch=epoch)]
    order = np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(len(identities))
    batches=[]; cursor=0
    for n in sizes:
        batches.append(tuple(map(int,order[cursor:cursor+n])));cursor+=n
    return batches


def validate_factor_permission(study, settings, variant):
    if settings.factor_arm is None:return
    permission=study.get('validation_factors',{})
    if (permission.get('authorized') is not True or permission.get('test_authorized') is not False
            or permission.get('arms') != list(ARMS) or settings.seed not in permission.get('seeds',[])):
        raise ValueError('single-factor study permission missing or mismatched')
    if settings.stage=='baseline' and variant != settings.method+'-'+settings.factor_arm:
        raise ValueError('single-factor arm identity mismatch')


def validate_terminal_binding(record, manifest, status, summary):
    from experiment_artifacts import manifest_sha256
    if (record['manifest_sha256'] != manifest_sha256(manifest)
            or record['run_id'] != manifest['run_id'] or summary['run_id'] != record['run_id']
            or summary['manifest_sha256'] != record['manifest_sha256']
            or status['state'] != record['event'] or summary['state'] != record['event']
            or summary['output_hashes'] != status['output_hashes']):
        raise ValueError('factor terminal identity/inventory mismatch')


def validate_factor_slot(completed_keys, key, limit):
    if type(limit) is not int or limit < 1 or key in completed_keys or len(completed_keys)>=limit:
        raise ValueError('duplicate factor run or scientific run limit exceeded')
