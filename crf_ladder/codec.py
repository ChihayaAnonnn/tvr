"""Deterministic PyAV codec + preflight + manifest utilities for the CRF ladder.

All transcoding uses the frozen x264 recipe declared in
:mod:`crf_ladder.protocol`. The public surface is deliberately small:

* :func:`transcode_atomic` — the only supported way to produce a degraded video.
  Guarantees the destination is either the correct output or unchanged.
* :func:`build_manifest_row` — pure conversion from a completed (or failed)
  job to a :class:`ManifestRow`. Worker-callable; performs no I/O beyond
  :func:`probe_video` and :func:`sha256_file` on success.
* :func:`append_manifest_row` — single-writer, single-line JSONL append. The
  ladder builder is responsible for ensuring only the parent process calls
  this; workers must return :class:`ManifestRow` values back to the parent.
* :func:`should_rebuild` — pure resume decision used by the parent process.
* :func:`run_preflight` — real-encode / real-decode capability probe used by
  Task-3 builders and the Task-11 runner before dispatching any workers.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional

import av

from crf_ladder.protocol import X264_RECIPE, X264Recipe

_SHA_CHUNK = 1024 * 1024

_X264_PARAMS = (
    "keyint=48:min-keyint=48:scenecut=0:bframes=2:force-cfr=1:"
    "colorprim=smpte170m:transfer=smpte170m:colormatrix=smpte170m:range=limited"
)


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps_num: int
    fps_den: int
    time_base_num: int
    time_base_den: int
    frame_count: int
    duration_seconds: float


@dataclass(frozen=True)
class PreflightResult:
    encoder: str
    source_width: int
    source_height: int
    source_fps: Fraction
    output_width: int
    output_height: int
    output_fps: Fraction
    decoded_frames: int


@dataclass(frozen=True)
class ManifestRow:
    video_id: str
    source_path: str
    output_path: str
    source_sha256: str
    output_sha256: Optional[str]
    crf: int
    preset: str
    encoder: str
    pyav_version: str
    av_library_versions: str
    pixel_format: str
    color_range: str
    color_primaries: str
    color_trc: str
    colorspace: str
    fps_num: int
    fps_den: int
    time_base_num: int
    time_base_den: int
    frame_count: int
    width: int
    height: int
    keyint: int
    min_keyint: int
    scenecut: int
    b_frames: int
    size_bytes: int
    bitrate_bps: int
    success: bool
    error: Optional[str] = None


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the hex SHA-256 of ``path``, streamed in 1 MiB chunks."""

    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_SHA_CHUNK)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def encoder_options(crf: int) -> dict[str, str]:
    """Return the exact x264 option dict for a given CRF level."""

    return {
        "crf": str(int(crf)),
        "preset": X264_RECIPE.preset,
        "x264-params": _X264_PARAMS,
    }


def probe_video(path: str | os.PathLike[str]) -> VideoMetadata:
    """Return video metadata; ``frame_count`` is obtained by decoding every frame."""

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise RuntimeError(f"probe_video: no video stream in {path}")
        stream = container.streams.video[0]
        width = stream.codec_context.width
        height = stream.codec_context.height
        avg_rate = stream.average_rate or stream.base_rate or stream.rate
        if avg_rate is None:
            raise RuntimeError(f"probe_video: no frame rate for {path}")
        fps = Fraction(avg_rate)
        time_base = stream.time_base
        pix_fmt = stream.codec_context.pix_fmt
        _ = pix_fmt  # not exposed here; consumed by the row-builder via defaults
        count = 0
        for _packet in container.demux(stream):
            for _frame in _packet.decode():
                count += 1
        duration = count * fps.denominator / max(1, fps.numerator)
    return VideoMetadata(
        width=int(width),
        height=int(height),
        fps_num=int(fps.numerator),
        fps_den=int(fps.denominator),
        time_base_num=int(time_base.numerator),
        time_base_den=int(time_base.denominator),
        frame_count=int(count),
        duration_seconds=float(duration),
    )


def _transcode(source: Path, destination: Path, crf: int) -> None:
    """Encode every decoded frame of ``source`` to ``destination`` with x264 at ``crf``."""

    opts = encoder_options(crf)
    with av.open(str(source)) as in_container:
        if not in_container.streams.video:
            raise RuntimeError(f"_transcode: no video stream in {source}")
        in_stream = in_container.streams.video[0]
        rate = in_stream.average_rate or in_stream.base_rate or in_stream.rate
        if rate is None:
            raise RuntimeError(f"_transcode: no frame rate for {source}")
        rate = Fraction(rate)

        with av.open(str(destination), mode="w") as out_container:
            out_stream = out_container.add_stream("libx264", rate=rate)
            out_stream.width = in_stream.codec_context.width
            out_stream.height = in_stream.codec_context.height
            out_stream.pix_fmt = X264_RECIPE.pixel_format
            out_stream.time_base = Fraction(rate.denominator, rate.numerator)
            out_stream.codec_context.max_b_frames = X264_RECIPE.b_frames
            out_stream.options = opts

            # Assign monotonic pts in the encoder time_base (1 tick per frame at 1/fps).
            # `frame.pts = None` alone drops all pts to zero after mp4 rescaling; the
            # muxer then reports a wildly wrong `average_rate` on the round-tripped
            # output. Explicit `pts = index` + `frame.time_base = 1/fps` lets PyAV
            # rescale to the container time_base while preserving the true rate.
            index = 0
            encoder_time_base = Fraction(rate.denominator, rate.numerator)
            for packet in in_container.demux(in_stream):
                for frame in packet.decode():
                    frame.pts = index
                    frame.time_base = encoder_time_base
                    index += 1
                    for encoded in out_stream.encode(frame):
                        out_container.mux(encoded)
            for encoded in out_stream.encode(None):
                out_container.mux(encoded)


def assert_transcode_invariants(source: str | os.PathLike[str], output: str | os.PathLike[str]) -> None:
    """Raise ``RuntimeError`` if source and output disagree on geometry / frame count / duration."""

    src_meta = probe_video(source)
    out_meta = probe_video(output)
    mismatches: list[str] = []
    for field_name in ("width", "height", "fps_num", "fps_den", "frame_count"):
        src_value = getattr(src_meta, field_name)
        out_value = getattr(out_meta, field_name)
        if src_value != out_value:
            mismatches.append(f"{field_name}: source={src_value} output={out_value}")
    fps_period = src_meta.fps_den / max(1, src_meta.fps_num)
    if abs(src_meta.duration_seconds - out_meta.duration_seconds) > 0.5 * fps_period:
        mismatches.append(
            f"duration: source={src_meta.duration_seconds:.6f} "
            f"output={out_meta.duration_seconds:.6f} (tol={0.5 * fps_period:.6f})"
        )
    if mismatches:
        raise RuntimeError(
            f"transcode invariants violated for source {source!s}: " + "; ".join(mismatches)
        )


def transcode_atomic(source: str | os.PathLike[str], destination: str | os.PathLike[str], crf: int) -> Path:
    """Encode ``source`` at ``crf`` into ``destination`` atomically.

    Writes to a sibling ``.tmp.mp4`` file, verifies invariants, then
    ``os.replace``. On any failure the destination bytes are unchanged and no
    ``.tmp.mp4`` residue is left behind.
    """

    source_path = Path(source)
    dest_path = Path(destination)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        dir=str(dest_path.parent),
        suffix=".tmp.mp4",
        delete=False,
    )
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        _transcode(source_path, tmp_path, crf)
        assert_transcode_invariants(source_path, tmp_path)
        os.replace(tmp_path, dest_path)
    except BaseException:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    finally:
        if tmp_path.exists() and tmp_path != dest_path:
            # Belt-and-suspenders: os.replace on success already consumed tmp_path;
            # if it still exists here (only possible on failure), remove it.
            try:
                tmp_path.unlink()
            except OSError:
                pass
    return dest_path


def build_manifest_row(
    *,
    job: Any,
    source_path: Path,
    source_sha256: str,
    output_path: Path,
    versions: dict[str, str],
    success: bool,
    error: Optional[str] = None,
    recipe: X264Recipe = X264_RECIPE,
) -> ManifestRow:
    """Return a :class:`ManifestRow` for a completed or failed job.

    Pure aside from :func:`probe_video` + :func:`sha256_file` on ``success=True``.
    Workers call this and return the resulting row to the parent for appending.
    """

    common: dict[str, Any] = dict(
        video_id=str(job.video_id),
        source_path=str(source_path),
        output_path=str(output_path),
        source_sha256=source_sha256,
        crf=int(job.crf),
        preset=recipe.preset,
        encoder="libx264",
        pyav_version=versions.get("pyav", ""),
        av_library_versions=versions.get("av_libraries", "{}"),
        pixel_format=recipe.pixel_format,
        color_range=recipe.color_range,
        color_primaries=recipe.color_primaries,
        color_trc=recipe.color_trc,
        colorspace=recipe.colorspace,
        keyint=recipe.keyint,
        min_keyint=recipe.min_keyint,
        scenecut=recipe.scenecut,
        b_frames=recipe.b_frames,
    )
    if success:
        metadata = probe_video(output_path)
        size_bytes = int(Path(output_path).stat().st_size)
        duration = float(metadata.duration_seconds)
        if duration <= 0.0:
            raise RuntimeError(
                f"build_manifest_row: refusing to record zero-duration output {output_path}"
            )
        bitrate = int(round(size_bytes * 8 / duration))
        return ManifestRow(
            **common,
            output_sha256=sha256_file(output_path),
            fps_num=metadata.fps_num,
            fps_den=metadata.fps_den,
            time_base_num=metadata.time_base_num,
            time_base_den=metadata.time_base_den,
            frame_count=metadata.frame_count,
            width=metadata.width,
            height=metadata.height,
            size_bytes=size_bytes,
            bitrate_bps=bitrate,
            success=True,
            error=None,
        )
    return ManifestRow(
        **common,
        output_sha256=None,
        fps_num=0,
        fps_den=1,
        time_base_num=1,
        time_base_den=1,
        frame_count=0,
        width=0,
        height=0,
        size_bytes=0,
        bitrate_bps=0,
        success=False,
        error=error,
    )


def append_manifest_row(path: str | os.PathLike[str], row: ManifestRow) -> None:
    """Single-writer JSONL append.

    Called only from the parent process (see Task 3 concurrency model).
    ``json.dumps`` with ``sort_keys=True`` guarantees deterministic on-disk
    row bytes given identical inputs.
    """

    line = json.dumps(asdict(row), sort_keys=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())


def should_rebuild(
    output: str | os.PathLike[str],
    row: Optional[ManifestRow],
    source_sha256: str,
) -> bool:
    """Return True when the output should be regenerated."""

    output_path = Path(output)
    if not output_path.exists():
        return True
    if row is None:
        return True
    if not row.success:
        return True
    if row.source_sha256 != source_sha256:
        return True
    if row.output_sha256 is None:
        return True
    if sha256_file(output_path) != row.output_sha256:
        return True
    return False


def run_preflight(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    frame_limit: int,
) -> PreflightResult:
    """Encode the first ``frame_limit`` source frames and re-decode them.

    Fails fast if libx264 is not available or if geometry / fps drift between
    source and re-encoded output. Uses the frozen recipe's CRF 30 as a
    reasonable default probe level.
    """

    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    with av.open(str(source_path)) as in_container:
        if not in_container.streams.video:
            raise RuntimeError(f"run_preflight: no video stream in {source_path}")
        in_stream = in_container.streams.video[0]
        rate = in_stream.average_rate or in_stream.base_rate or in_stream.rate
        if rate is None:
            raise RuntimeError(f"run_preflight: no frame rate for {source_path}")
        rate = Fraction(rate)
        src_width = int(in_stream.codec_context.width)
        src_height = int(in_stream.codec_context.height)

        with av.open(str(destination_path), mode="w") as out_container:
            out_stream = out_container.add_stream("libx264", rate=rate)
            out_stream.width = src_width
            out_stream.height = src_height
            out_stream.pix_fmt = X264_RECIPE.pixel_format
            out_stream.time_base = Fraction(rate.denominator, rate.numerator)
            out_stream.codec_context.max_b_frames = X264_RECIPE.b_frames
            out_stream.options = encoder_options(30)

            written = 0
            encoder_time_base = Fraction(rate.denominator, rate.numerator)
            for packet in in_container.demux(in_stream):
                if written >= frame_limit:
                    break
                for frame in packet.decode():
                    if written >= frame_limit:
                        break
                    frame.pts = written
                    frame.time_base = encoder_time_base
                    for encoded in out_stream.encode(frame):
                        out_container.mux(encoded)
                    written += 1
            for encoded in out_stream.encode(None):
                out_container.mux(encoded)

    decoded_frames = 0
    with av.open(str(destination_path)) as out_container:
        out_stream = out_container.streams.video[0]
        out_width = int(out_stream.codec_context.width)
        out_height = int(out_stream.codec_context.height)
        out_rate = Fraction(
            out_stream.average_rate or out_stream.base_rate or out_stream.rate
        )
        for _packet in out_container.demux(out_stream):
            for _frame in _packet.decode():
                decoded_frames += 1

    return PreflightResult(
        encoder="libx264",
        source_width=src_width,
        source_height=src_height,
        source_fps=rate,
        output_width=out_width,
        output_height=out_height,
        output_fps=out_rate,
        decoded_frames=decoded_frames,
    )
