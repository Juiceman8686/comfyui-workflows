"""Streaming stages around DLSS5: an RTX/RIFE pre-pass before it and RTX Video HDR after it.

Pre-pass, in this order, per frame, written once to a near-lossless intermediate:
  1. RTX Video Super Resolution clean pass (DEBLUR_*/DENOISE_*) at source resolution
  2. RIFE 4.9 up to the target frame rate (timestamp-based, so any source rate lands on the grid;
     pairs that straddle a hard cut are held, not morphed)
  3. optional RTX VSR pre-upscale (HIGHBITRATE_* etc.) so DLSS5 can run at a lower factor

The order comes from measurements (LPIPS against ground truth plus crops of real AI footage):
cleaning before DLSS5 helped and cleaning after it did not, DENOISE before DLSS5 always lost
texture DLSS5 could not rebuild, and DEBLUR_ULTRA at strength 0.5 was the best all-rounder.
Cleaning first also means RIFE estimates motion on sharper frames and the clean pass runs on the
fewest, smallest frames.

Post-pass: RTX Video HDR on the DLSS5 output, encoded as 10-bit HEVC tagged BT.2020 HLG or PQ.
It has to come last: DLSS5 works in 8-bit SDR and would clip HDR values.

Nothing is ever held as a whole clip; frames stream through, so long videos are fine.
RIFE weights come from comfyui-frame-interpolation, the RTX effects from nvidia-vfx (nvvfx).
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import logging
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import folder_paths
import nodes
import numpy as np
import torch
from comfy.model_management import get_torch_device, throw_exception_if_processing_interrupted
from comfy.utils import ProgressBar

logger = logging.getLogger(__name__)

TEMP_PREFIX = "DLSS5_pre_"
_RIFE_DIR = Path(folder_paths.base_path) / "custom_nodes" / "comfyui-frame-interpolation"
_RIFE_CKPT = _RIFE_DIR / "ckpts" / "rife" / "rife49.pth"
_RIFE_ARCH_VER = "4.7"  # rife49.pth uses the 4.7 architecture
# Below this a target frame is close enough to a real one to just reuse it.
_SNAP = 0.02

CLEAN_MODES = ["off"] + [f"{kind}_{level}" for kind in ("DEBLUR", "DENOISE") for level in ("ULTRA", "HIGH", "MEDIUM", "LOW")]
UPSCALE_QUALITIES = [
    "HIGHBITRATE_ULTRA", "HIGHBITRATE_HIGH", "HIGHBITRATE_MEDIUM", "HIGHBITRATE_LOW",
    "ULTRA", "HIGH", "MEDIUM", "LOW", "STREAMING_ULTRA", "STREAMING_MEDIUM", "BICUBIC",
]
UPSCALE_FACTORS = {"off": 1.0, "1.5x": 1.5, "2x": 2.0}

_model = None


@dataclass(frozen=True)
class PreOptions:
    clean_mode: str = "DEBLUR_ULTRA"
    clean_strength: float = 0.5
    upscale: str = "off"
    upscale_quality: str = "HIGHBITRATE_ULTRA"
    interpolate: bool = False
    target_fps: float = 60.0
    cut_threshold: float = 0.30

    @property
    def cleans(self) -> bool:
        return self.clean_mode != "off" and self.clean_strength > 0


@dataclass(frozen=True)
class HDROptions:
    enabled: bool = False
    colorspace: str = "HLG"
    contrast: int = 100
    saturation: int = 100
    middle_gray: int = 50
    luminance: int = 650
    debanding: bool = True
    keep_sdr: bool = True


def _pack_module(node_id: str, pack: str, *names: str):
    """Packs load under path-derived module names; reach their modules through a node class."""
    node = nodes.NODE_CLASS_MAPPINGS.get(node_id)
    if node is None:
        raise RuntimeError(f"{node_id} is not loaded; is {pack} installed?")
    return [importlib.import_module(f"{node.__module__}{name}") for name in names]


def _dlss5_modules():
    enhancer = nodes.NODE_CLASS_MAPPINGS.get("DLSS5EnhanceVideoFile")
    if enhancer is None:
        raise RuntimeError("DLSS5 Enhance Video File is not loaded; is ComfyUI-DLSS5-Enhancer installed?")
    base = enhancer.__module__.rsplit(".nodes.", 1)[0]
    return importlib.import_module(f"{base}.dlss5.media"), importlib.import_module(f"{base}.nodes.common")


def _rtx_module():
    """The NVIDIA VFX pack's own helpers (RGB10A2 unpacking, PQ -> HLG)."""
    return _pack_module("RTXTrueHDR", "comfyui_nvidia_rtx_nodes", "")[0]


def _load_rife():
    global _model
    if _model is None:
        if not _RIFE_CKPT.is_file():
            raise FileNotFoundError(
                f"RIFE weights not found at {_RIFE_CKPT}. Install comfyui-frame-interpolation and run "
                "its RIFE VFI node once so it downloads rife49.pth."
            )
        spec = importlib.util.spec_from_file_location(
            "dlss5_helpers_rife_arch", _RIFE_DIR / "vfi_models" / "rife" / "rife_arch.py"
        )
        arch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(arch)
        model = arch.IFNet(arch_ver=_RIFE_ARCH_VER)
        model.load_state_dict(torch.load(_RIFE_CKPT, map_location="cpu", weights_only=False))
        _model = model.eval().to(get_torch_device()).half()
    return _model


def _cut_features(rgb: np.ndarray) -> np.ndarray:
    """HSV histogram of a small copy; its distance separates hard cuts from fast motion."""
    scale = 320 / rgb.shape[1]
    small = cv2.resize(rgb, (320, max(1, int(rgb.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    hist = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_RGB2HSV)], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def clean_temp() -> None:
    """Remove intermediates from earlier runs; each one is only needed within its own prompt."""
    for stale in Path(folder_paths.get_temp_directory()).glob(f"{TEMP_PREFIX}*.mov"):
        try:
            stale.unlink()
        except OSError:
            pass


def _vsr(stack: contextlib.ExitStack, quality: str, strength: float, width: int, height: int):
    """Open an RTX VSR effect; returns (run(float CHW) -> float CHW, out_w, out_h)."""
    import nvvfx

    device = get_torch_device()
    effect = stack.enter_context(nvvfx.VideoSuperRes(
        quality=nvvfx.QualityLevel[quality], strength=strength,
        device=device.index if device.index is not None else torch.cuda.current_device(),
        image_encoding=nvvfx.ImageEncoding.RGB8,
    ))
    effect.output_width, effect.output_height = width, height
    effect.load()

    def run(frame: torch.Tensor) -> torch.Tensor:
        return torch.from_dlpack(effect.run(frame.float().contiguous()).image).clone().clamp_(0, 1)

    return run


def preprocess(source: Path, settings, opts: PreOptions, limit_seconds: float = 0) -> Path | None:
    """Run the pre-pass on ``source`` into a temp MOV, or return None when it has nothing to do.

    ``limit_seconds`` > 0 only processes the start of the clip (quick preview)."""
    media, common = _dlss5_modules()
    layout = common.resolve_layout(settings)
    layout.require_ffmpeg()
    meta = media.probe_video(layout, source)
    interpolate = opts.interpolate and meta["fps"] < opts.target_fps - 0.01
    if opts.interpolate and not interpolate:
        logger.info("DLSS5 pre-pass: %s is already %.3f fps, not interpolating", source.name, meta["fps"])
    factor = UPSCALE_FACTORS[opts.upscale]
    if not (opts.cleans or interpolate or factor > 1):
        return None

    width, height, rotation = meta["width"], meta["height"], meta["rotation"]
    if width % 2 or height % 2:
        raise ValueError(f"{source.name} is {width}x{height}; the pre-pass needs even dimensions.")
    # RTX VSR rounds upscaled sizes to multiples of 8.
    out_w, out_h = (max(8, round(width * factor / 8) * 8), max(8, round(height * factor / 8) * 8)) if factor > 1 else (width, height)
    rate = opts.target_fps if interpolate else float(meta["rate"])
    rate_arg = f"{opts.target_fps:g}" if interpolate else f"{meta['rate'].numerator}/{meta['rate'].denominator}"
    duration = meta["duration"] or (meta["frames"] / meta["fps"] if meta["fps"] else 0)
    if limit_seconds > 0:
        duration = min(duration, limit_seconds) if duration else limit_seconds
    expected = max(1, int(math.floor(duration * (opts.target_fps if interpolate else meta["fps"]))))
    # RIFE recommends half-scale flow above roughly 1080p.
    flow_scale = 0.5 if width * height > 2304 * 1296 else 1.0
    scale_list = [8 / flow_scale, 4 / flow_scale, 2 / flow_scale, 1 / flow_scale]

    out_path = Path(folder_paths.get_temp_directory()) / f"{TEMP_PREFIX}{int(time.time() * 1000)}.mov"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def open_encoder(first_time: float) -> subprocess.Popen:
        command = [
            str(layout.ffmpeg), "-v", "error", "-y",
            # Offset the new video by the source's first frame time so audio stays in sync.
            "-itsoffset", f"{first_time:.6f}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{out_w}x{out_h}", "-framerate", rate_arg, "-i", "-",
            "-i", str(source),
            "-map", "0:v", "-map", "1:a?", "-map_metadata", "1", "-map_chapters", "1",
            # Near-lossless intermediate; DLSS5 re-encodes the final file anyway. 4:4:4 because the
            # extra RGB -> 4:2:0 -> RGB round trip measurably darkened every frame by ~1 level.
            "-c:v", "hevc_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "constqp", "-qp", "12", "-pix_fmt", "yuv444p",
            # MOV records a frame count (MKV would force DLSS5 into a slow counting pass), and PCM
            # holds any source audio codec without loss.
            "-c:a", "pcm_s24le",
            # A preview keeps only as much audio as video.
            *(["-t", f"{first_time + limit_seconds:.6f}"] if limit_seconds > 0 else []),
            str(out_path),
        ]
        return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    device = get_torch_device()
    progress = ProgressBar(expected)
    encoder = None
    written = cuts = 0

    with contextlib.ExitStack() as stack:
        clean = _vsr(stack, opts.clean_mode, opts.clean_strength, width, height) if opts.cleans else None
        upscale = _vsr(stack, opts.upscale_quality, 1.0, out_w, out_h) if factor > 1 else None
        model = _load_rife() if interpolate else None

        def to_gpu(rgb: np.ndarray) -> torch.Tensor:
            frame = torch.from_numpy(rgb).to(device).permute(2, 0, 1).float().div_(255)
            if clean is not None:
                frame = clean(frame)
            return frame[None].half()

        def emit(tensor: torch.Tensor) -> None:
            nonlocal written
            frame = tensor[0, :, :height, :width]
            if upscale is not None:
                frame = upscale(frame)
            frame = frame.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0)
            encoder.stdin.write(np.ascontiguousarray(frame.cpu().numpy()).tobytes())
            written += 1
            progress.update(1)

        try:
            prev = None  # (time, gpu tensor, cut features)
            start = None
            with torch.inference_mode():
                for _, rgba, pts in media.decode_frames(source, rotation):
                    throw_exception_if_processing_interrupted()
                    rgb = np.ascontiguousarray(rgba[..., :3])
                    now = float(pts * meta["time_base"])
                    if encoder is None:
                        start = now
                        encoder = open_encoder(max(0.0, start))
                    elif limit_seconds > 0 and now - start >= limit_seconds:
                        break
                    if not interpolate:
                        emit(to_gpu(rgb))
                        continue
                    current = (now, to_gpu(rgb), _cut_features(rgb))
                    if prev is None:
                        prev = current
                        continue
                    t0, t1 = prev[0], now
                    if t1 <= t0:  # duplicate or out-of-order timestamp
                        prev = current
                        continue
                    is_cut = cv2.compareHist(prev[2], current[2], cv2.HISTCMP_BHATTACHARYYA) > opts.cut_threshold
                    cuts += is_cut
                    # Output slots that fall between these two real frames, as fractions of the gap.
                    wanted, slot = [], written
                    while start + slot / opts.target_fps < t1 - 1e-6:
                        wanted.append((start + slot / opts.target_fps - t0) / (t1 - t0))
                        slot += 1
                    steps = [s for s in wanted if _SNAP < s < 1 - _SNAP and not is_cut]
                    middles = iter(())
                    if steps:
                        count = len(steps)
                        timestep = torch.tensor(steps, device=device, dtype=torch.half).view(-1, 1, 1, 1)
                        middles = iter(model(
                            prev[1].expand(count, -1, -1, -1), current[1].expand(count, -1, -1, -1),
                            timestep, scale_list, training=False, fastmode=True, ensemble=False,
                        ).split(1))
                    for step in wanted:
                        if step <= _SNAP or (is_cut and step < 0.5):
                            emit(prev[1])
                        elif step >= 1 - _SNAP or is_cut:
                            emit(current[1])
                        else:
                            emit(next(middles))
                    prev = current
                if encoder is None:
                    raise RuntimeError(f"No frames could be decoded from {source}.")
                if interpolate:
                    # Close out the grid up to the last real frame.
                    while start + written / opts.target_fps <= prev[0] + 0.5 / opts.target_fps:
                        emit(prev[1])
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("ffmpeg failed writing the pre-pass intermediate: "
                                   f"{encoder.stderr.read().decode(errors='replace')[-2000:]}")
        except BaseException:
            if encoder is not None and encoder.poll() is None:
                encoder.kill()
                encoder.wait()
            out_path.unlink(missing_ok=True)
            raise
        finally:
            torch.cuda.empty_cache()
    logger.info(
        "DLSS5 pre-pass: %s -> %dx%d @ %.3f fps, %d frames (clean %s @ %.2f, upscale %s, %d cut(s) held)",
        source.name, out_w, out_h, rate, written,
        opts.clean_mode if opts.cleans else "off", opts.clean_strength, opts.upscale, cuts,
    )
    return out_path


# BT.2020 non-constant-luminance RGB -> Y'CbCr, applied to the PQ/HLG-encoded signal.
_KR, _KB = 0.2627, 0.0593
_KG = 1 - _KR - _KB


def _to_yuv420p10(rgb: torch.Tensor) -> bytes:
    """(3, H, W) float 0..1 BT.2020 signal -> limited-range yuv420p10le planes."""
    r, g, b = rgb[0], rgb[1], rgb[2]
    y = _KR * r + _KG * g + _KB * b
    cb = (b - y) / (2 * (1 - _KB))
    cr = (r - y) / (2 * (1 - _KR))
    pool = lambda c: torch.nn.functional.avg_pool2d(c[None, None], 2)[0, 0]
    planes = (
        (64 + 876 * y).round().clamp(0, 1023),
        (512 + 896 * pool(cb)).round().clamp(0, 1023),
        (512 + 896 * pool(cr)).round().clamp(0, 1023),
    )
    return b"".join(p.to(torch.int16).cpu().numpy().astype("<u2").tobytes() for p in planes)


def apply_hdr(source: Path, settings, opts: HDROptions, target: Path) -> Path:
    """Convert a finished SDR render to 10-bit HDR (HLG or PQ) with RTX Video HDR."""
    import nvvfx

    media, common = _dlss5_modules()
    rtx = _rtx_module()
    layout = common.resolve_layout(settings)
    layout.require_ffmpeg()
    meta = media.probe_video(layout, source)
    width, height = meta["width"], meta["height"]
    if width % 2 or height % 2:
        raise ValueError(f"{source.name} is {width}x{height}; 4:2:0 HDR needs even dimensions.")
    trc = "arib-std-b67" if opts.colorspace == "HLG" else "smpte2084"
    bitrate = media.auto_bitrate_kbps(width, height, meta["fps"], "HEVC") * 4  # DLSS5's "Best"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.stem}_partial{target.suffix}")
    command = [
        str(layout.ffmpeg), "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-s", f"{width}x{height}", "-framerate", f"{meta['rate'].numerator}/{meta['rate'].denominator}",
        "-color_primaries", "bt2020", "-color_trc", trc, "-colorspace", "bt2020nc", "-color_range", "tv", "-i", "-",
        "-i", str(source),
        "-map", "0:v", "-map", "1:a?", "-map_metadata", "1", "-map_chapters", "1",
        "-c:v", "hevc_nvenc", "-preset", "p6", "-tune", "hq", "-profile:v", "main10", "-rc", "vbr", "-b:v", f"{bitrate}k",
        "-pix_fmt", "p010le", "-color_primaries", "bt2020", "-color_trc", trc, "-colorspace", "bt2020nc", "-color_range", "tv",
        "-c:a", "copy", *(["-tag:v", "hvc1"] if target.suffix.lower() in (".mp4", ".mov") else []),
        str(partial),
    ]
    device = get_torch_device()
    progress = ProgressBar(max(1, meta["frames"]))
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    written = 0
    try:
        with torch.cuda.device(device), nvvfx.TrueHDR(
            contrast=opts.contrast, saturation=opts.saturation, middle_gray=opts.middle_gray,
            luminance=opts.luminance, debanding_off=int(not opts.debanding),
            device=device.index if device.index is not None else torch.cuda.current_device(),
        ) as effect, torch.inference_mode():
            effect.load()
            for _, rgba, _ in media.decode_frames(source, 0):
                throw_exception_if_processing_interrupted()
                frame = torch.from_numpy(np.ascontiguousarray(rgba[..., :3])).to(device).permute(2, 0, 1).float().div_(255)
                hdr = rtx.unpack_rgb10a2(torch.from_dlpack(effect.run(frame.contiguous()).image).clone())  # (H, W, 3) PQ
                if opts.colorspace == "HLG":
                    hdr = rtx.pq_to_hlg(hdr, opts.luminance)
                encoder.stdin.write(_to_yuv420p10(hdr.permute(2, 0, 1)))
                written += 1
                progress.update(1)
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing the HDR file: {encoder.stderr.read().decode(errors='replace')[-2000:]}")
        partial.replace(target)
    except BaseException:
        if encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        partial.unlink(missing_ok=True)
        raise
    finally:
        torch.cuda.empty_cache()
    logger.info("DLSS5 HDR: %s -> %s (%s, %d nits, %d frames)", source.name, target.name, opts.colorspace, opts.luminance, written)
    return target
