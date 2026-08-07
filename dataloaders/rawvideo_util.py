import os

import cv2
import numpy as np
import torch as th
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

# pytorch=1.7.1
# pip install opencv-python

# CLIP-style ImageNet-ish normalization stats (kept as module-level tensors so we
# can apply them once to a stacked tensor instead of per-frame inside `Compose`).
_CLIP_MEAN = th.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD = th.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

try:
    import decord as _decord  # noqa: F401

    _DECORD_AVAILABLE = True
except Exception:
    _DECORD_AVAILABLE = False


def _compute_target_frames(start_sec, end_sec, fps, inds):
    """Materialize the sorted, deduplicated absolute frame indices to sample.

    The old implementation seeked to each of these positions individually,
    which is catastrophic on inter-frame-coded video (H.264/H.265): every
    `cap.set(CAP_PROP_POS_FRAMES, N)` forces OpenCV/FFmpeg to jump back to the
    nearest keyframe and re-decode the intervening P/B frames. We now compute
    the whole target set up front and stream through the file once.
    """
    targets = []
    for sec in range(start_sec, end_sec + 1):
        sec_base = sec * fps
        for ind in inds:
            targets.append(sec_base + int(ind))
    # Preserve sampling order for downstream models but drop any duplicates
    # that could arise from tiny videos with `inds == [0]` at second boundaries.
    seen = set()
    ordered = []
    for idx in targets:
        if idx in seen:
            continue
        seen.add(idx)
        ordered.append(idx)
    ordered.sort()
    return ordered


class RawVideoExtractorCV2():
    def __init__(self, centercrop=False, size=224, framerate=-1, ):
        if hasattr(cv2, "setNumThreads"):
            cv2.setNumThreads(1)
        self.centercrop = centercrop
        self.size = size
        self.framerate = int(framerate) if framerate > 0 else framerate
        # Per-frame geometry (Resize+CenterCrop+RGB+ToTensor). Normalize is
        # applied once to the stacked tensor in `_normalize_stack` — this is
        # numerically identical but avoids N tiny CPU ops.
        self.transform = self._transform(self.size)


    def _transform(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
        ])

    @staticmethod
    def _normalize_stack(tensor_nchw):
        """Fused CLIP normalization applied to a whole [N, 3, H, W] batch."""
        return tensor_nchw.sub_(_CLIP_MEAN).div_(_CLIP_STD)

    def video_to_tensor(self, video_file, preprocess, sample_fp=0, start_time=None, end_time=None):
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                   and start_time > -1 and end_time > start_time
        assert sample_fp > -1

        cap = cv2.VideoCapture(video_file)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        total_duration = (frameCount + fps - 1) // fps
        start_sec, end_sec = 0, total_duration

        if start_time is not None:
            start_sec, end_sec = start_time, end_time if end_time <= total_duration else total_duration

        interval = 1
        if sample_fp > 0:
            interval = fps // sample_fp
        else:
            sample_fp = fps
        if interval == 0:
            interval = 1

        inds = [int(ind) for ind in range(0, fps, interval)]
        assert len(inds) >= sample_fp
        inds = inds[:sample_fp]

        targets = _compute_target_frames(start_sec, end_sec, fps, inds)

        images = self._decode_targets(
            cap, targets, preprocess=preprocess, want_bgr=False
        )
        cap.release()

        if len(images) > 0:
            # `preprocess` returns a [3, H, W] tensor per frame (no Normalize).
            video_data = th.stack(images).contiguous()
            video_data = self._normalize_stack(video_data)
        else:
            video_data = th.zeros(1)
        return {'video': video_data}

    def _decode_targets(self, cap, targets, *, preprocess, want_bgr):
        """Sequentially walk the stream and emit only frames at `targets`.

        - `cap.grab()` advances the decoder without copying the frame out to
          Python — much cheaper than `read()` when we're just skipping.
        - `cap.read()` = grab + retrieve — used only when we've landed on a
          target index.
        - We seek exactly once, and only if the first target is far enough
          into the file that a keyframe rewind beats grabbing sequentially.
          Otherwise we stream from position 0, which is how the decoder is
          happiest (no lost inter-frame prediction state).

        `cur` tracks the position of the NEXT frame the decoder will yield.
        Both `grab()` and `read()` advance `cur` by exactly one.
        """
        if not targets:
            return []

        cur = 0
        # Heuristic: one seek pays off once the jump is larger than about a
        # long GOP. Below that, grab() is cheaper than a keyframe rewind + the
        # forced sequential decode back up to the target.
        SEEK_THRESHOLD = 256
        first = targets[0]
        if first >= SEEK_THRESHOLD:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(first))
            reported = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            # Trust the backend's reported position when it looks sane;
            # otherwise fall back to streaming from 0.
            if 0 <= reported <= first:
                cur = reported

        out = []
        for next_target in targets:
            if next_target < cur:
                # Backend overshot the seek; the target is already behind us.
                # Skipping is the least-bad option — reversing would require
                # another full rewind.
                continue
            while cur < next_target:
                if not cap.grab():
                    return out
                cur += 1
            ok, frame = cap.read()
            if not ok or frame is None:
                return out
            cur += 1
            if want_bgr:
                out.append(frame)
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out.append(preprocess(Image.fromarray(frame_rgb)))

        return out

    def get_video_data(self, video_path, start_time=None, end_time=None):
        image_input = self.video_to_tensor(video_path, self.transform, sample_fp=self.framerate, start_time=start_time, end_time=end_time)
        return image_input

    def get_tqfs_video_data(self, video_path, num_frames, start_time=None, end_time=None):
        """Decode once, select TQFS frames, then preprocess only selected frames."""

        from dataloaders.tqfs_util import select_tqfs_indices

        raw_frames = self.get_raw_video_data(
            video_path, start_time=start_time, end_time=end_time
        )["video"]
        if not raw_frames:
            return {"video": th.zeros(1)}

        selected_indices = select_tqfs_indices(raw_frames, int(num_frames))
        selected_frames = [raw_frames[index] for index in selected_indices]
        video = self.preprocess_raw_frames(selected_frames).squeeze(1)
        return {"video": video}

    def process_raw_data(self, raw_video_data):
        tensor_size = raw_video_data.size()
        tensor = raw_video_data.view(-1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1])
        return tensor

    def get_raw_video_data(self, video_path, start_time=None, end_time=None):
        """Decode raw BGR frames without CLIP preprocessing.

        Used by TQFS-style paths that need to compute quality metrics on the
        original pixels before choosing which frames to keep. Returns a list
        rather than a stacked tensor so downstream cv2 calls (which are
        picky about numpy layout) work without a copy.
        """
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                   and start_time > -1 and end_time > start_time

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30

        total_duration = (frame_count + fps - 1) // fps
        start_sec, end_sec = 0, total_duration
        if start_time is not None:
            start_sec = start_time
            end_sec = min(end_time, total_duration)

        if self.framerate > 0:
            interval = max(1, fps // self.framerate)
            inds = list(range(0, fps, interval))[:self.framerate]
        else:
            interval = 1
            inds = list(range(0, fps))

        targets = _compute_target_frames(start_sec, end_sec, fps, inds)
        images = self._decode_targets(cap, targets, preprocess=None, want_bgr=True)
        cap.release()

        return {'video': images}

    def preprocess_raw_frames(self, raw_frames_list):
        """Apply CLIP preprocessing to a list of raw BGR frames.

        Returns a `[N, 1, 3, size, size]` float tensor (Normalize is fused on
        the stack for speed; result is bit-identical to per-frame Normalize).
        """
        frames = []
        for frame_bgr in raw_frames_list:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames.append(self.transform(pil_img))
        stacked = th.stack(frames).contiguous()
        stacked = self._normalize_stack(stacked)
        return stacked.unsqueeze(1)  # [N, 1, 3, size, size]

    def process_frame_order(self, raw_video_data, frame_order=0):
        # 0: ordinary order; 1: reverse order; 2: random order.
        if frame_order == 0:
            pass
        elif frame_order == 1:
            reverse_order = np.arange(raw_video_data.size(0) - 1, -1, -1)
            raw_video_data = raw_video_data[reverse_order, ...]
        elif frame_order == 2:
            random_order = np.arange(raw_video_data.size(0))
            np.random.shuffle(random_order)
            raw_video_data = raw_video_data[random_order, ...]

        return raw_video_data

class RawVideoExtractorDecord(RawVideoExtractorCV2):
    """Decord-backed extractor.

    Same interface as the CV2 version. Decord decodes in C++ with FFmpeg,
    exposes `get_batch(indices)` which internally sorts and GOP-batches
    the request, and returns numpy arrays without per-frame Python overhead.
    On our MSRVTT videos this typically beats the optimized-CV2 path 1.5-3x.

    NOTE ON REPRODUCIBILITY: decord's decode is bit-close but not
    bit-identical to OpenCV — different FFmpeg builds, different YUV→RGB
    routines, subpixel-level differences. Do NOT swap backends mid-experiment
    without re-verifying downstream metrics.
    """

    def __init__(self, centercrop=False, size=224, framerate=-1, num_threads=1):
        if not _DECORD_AVAILABLE:
            raise RuntimeError(
                "decord is not installed. `pip install decord==0.6.0` or fall "
                "back to the cv2 backend."
            )
        super().__init__(centercrop=centercrop, size=size, framerate=framerate)
        self._decord_num_threads = int(num_threads)

    def _open(self, video_path):
        # ctx=cpu(0): CPU decode. GPU decode is available (decord.gpu(id)) but
        # copies raw frames into GPU memory which we then have to fetch back
        # for PIL preprocessing — net loss for our pipeline.
        from decord import VideoReader, cpu  # local import to keep module load cheap

        return VideoReader(video_path, ctx=cpu(0), num_threads=self._decord_num_threads)

    def _compute_targets_for(self, vr, start_time, end_time, sample_fp):
        # Mirror the cv2-path index computation so both backends sample the
        # exact same absolute frame indices.
        frame_count = len(vr)
        fps = int(vr.get_avg_fps()) or 30
        total_duration = (frame_count + fps - 1) // fps
        start_sec, end_sec = 0, total_duration
        if start_time is not None:
            start_sec = start_time
            end_sec = min(end_time, total_duration)

        if sample_fp > 0:
            interval = max(1, fps // sample_fp)
            inds = list(range(0, fps, interval))[:sample_fp]
        else:
            inds = list(range(0, fps))

        targets = _compute_target_frames(start_sec, end_sec, fps, inds)
        # Clip to actual frame count — decord raises on out-of-range indices,
        # whereas cv2.read() silently returned False.
        targets = [t for t in targets if t < frame_count]
        return targets

    def video_to_tensor(self, video_file, preprocess, sample_fp=0,
                         start_time=None, end_time=None):
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                   and start_time > -1 and end_time > start_time
        assert sample_fp > -1

        vr = self._open(video_file)
        targets = self._compute_targets_for(vr, start_time, end_time, sample_fp)
        if not targets:
            return {'video': th.zeros(1)}

        # Single batched decode — decord sorts, groups by keyframe, and hands
        # us back a contiguous NDArray of RGB uint8.
        batch = vr.get_batch(targets).asnumpy()  # [N, H, W, 3] RGB

        frames = [preprocess(Image.fromarray(batch[i])) for i in range(batch.shape[0])]
        video_data = th.stack(frames).contiguous()
        video_data = self._normalize_stack(video_data)
        return {'video': video_data}

    def get_raw_video_data(self, video_path, start_time=None, end_time=None):
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                   and start_time > -1 and end_time > start_time

        vr = self._open(video_path)
        targets = self._compute_targets_for(vr, start_time, end_time, self.framerate)
        if not targets:
            return {'video': []}

        batch = vr.get_batch(targets).asnumpy()  # RGB
        # Downstream TQFS code expects BGR (channels 0=B, 1=G, 2=R). Copy so
        # each frame is independently contiguous — sklearn/KMeans and the
        # sharpness kernel walk arrays with numpy indexing that dislikes
        # negative strides.
        out = [np.ascontiguousarray(batch[i, :, :, ::-1]) for i in range(batch.shape[0])]
        return {'video': out}


def _resolve_backend():
    """Return the extractor class to use based on the TVR_VIDEO_BACKEND env var.

    - unset / `cv2` / `opencv` → optimized CV2 path (default; matches original
      pixel semantics exactly).
    - `decord` → decord path (faster, but pixel-close-not-identical to CV2).
    """
    choice = os.environ.get("TVR_VIDEO_BACKEND", "cv2").strip().lower()
    if choice in ("", "cv2", "opencv"):
        return RawVideoExtractorCV2
    if choice == "decord":
        if not _DECORD_AVAILABLE:
            raise RuntimeError(
                "TVR_VIDEO_BACKEND=decord but decord is not installed. "
                "Install with `pip install decord==0.6.0`."
            )
        return RawVideoExtractorDecord
    raise ValueError(f"Unknown TVR_VIDEO_BACKEND={choice!r}; use cv2 or decord")


def RawVideoExtractor(*args, **kwargs):
    """Factory that instantiates the extractor selected by TVR_VIDEO_BACKEND.

    Kept as a callable (not just an alias) so callers can flip the backend
    with an env var without touching import sites.
    """
    return _resolve_backend()(*args, **kwargs)
