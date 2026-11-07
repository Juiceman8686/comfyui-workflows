"""Local helpers around DLSS5 Enhance Video File: control panel, RTX/RIFE pre-pass, RTX HDR, batch.

The workflow is: Control Panel (toggles, no wires) -> Input (Load Video or Batch Folder) ->
settings for RTX Clean + 60 FPS, DLSS5 and HDR -> one Render node (single video) or Render Folder
node (batch). The render nodes stream the clip through pipeline.py's pre-pass, DLSS5 Enhance Video
File and the HDR post-pass without ever decoding it into an IMAGE batch. web/dlss5_panel.js wires
the panel's toggles to the graph: it mutes whichever input/render pair is not in use and greys out
settings for features that are off.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import folder_paths
import nodes
from comfy.model_management import InterruptProcessingException
from comfy_execution.graph_utils import ExecutionBlocker
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io, ui

from .pipeline import (
    CLEAN_MODES,
    UPSCALE_FACTORS,
    UPSCALE_QUALITIES,
    HDROptions,
    PreOptions,
    apply_hdr,
    clean_temp,
    preprocess as pre_pass,
)

WEB_DIRECTORY = "./web"

logger = logging.getLogger(__name__)

_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".ts"}
_CONTAINER_EXTENSIONS = {"MP4": ".mp4", "MKV": ".mkv", "MOV": ".mov"}
_PARTIAL_PREFIX = "DLSS5_batch_partial"
_RTX_CLEAN_MODES = [mode for mode in CLEAN_MODES if mode != "off"]

PRE_TYPE = io.Custom("DLSS5_PREPROCESS")
HDR_TYPE = io.Custom("DLSS5_HDR")
FOLDER_TYPE = io.Custom("DLSS5_FOLDER")
DLSS5_SETTINGS = io.Custom("DLSS5_SETTINGS")


def _resolve_folder(raw: str, root: str) -> Path:
    """An absolute path is used as given; a relative one is resolved inside ``root``."""
    value = raw.strip().strip('"').strip("'")
    if not value:
        raise ValueError("The folder path is empty.")
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else Path(root) / candidate).resolve()


def _enhancer():
    enhancer = nodes.NODE_CLASS_MAPPINGS.get("DLSS5EnhanceVideoFile")
    if enhancer is None:
        raise RuntimeError("DLSS5 Enhance Video File is not loaded; is ComfyUI-DLSS5-Enhancer installed?")
    return enhancer


def _preview_ui(path: Path):
    """A browser preview for files inside the output folder, otherwise the path as text."""
    path = path.resolve()
    output_root = Path(folder_paths.get_output_directory()).resolve()
    if not path.is_relative_to(output_root):
        return ui.PreviewText(str(path))
    subfolder = path.parent.relative_to(output_root).as_posix()
    return ui.PreviewVideo([ui.SavedResult(path.name, "" if subfolder == "." else subfolder, io.FolderType.output)])


def _render(source: Path, settings, preprocess: PreOptions, *, output_dir: Path, prefix: str, codec: str,
            container: str, quality: str, copy_audio: bool, verify: bool, preview_seconds: float) -> tuple[Path, int]:
    """STEP 2 pre-pass -> STEP 3 DLSS5 for one clip. Returns the SDR render and its frame count."""
    clean_temp()
    prepared = pre_pass(source, settings, preprocess, limit_seconds=preview_seconds)
    try:
        max_frames = 0
        if preview_seconds > 0 and prepared is None:
            # No pre-pass ran, so DLSS5 itself has to stop after the preview length.
            from .pipeline import _dlss5_modules

            media, common = _dlss5_modules()
            max_frames = max(1, math.ceil(preview_seconds * media.probe_video(common.resolve_layout(settings), source)["fps"]))
        result = _enhancer().execute(
            video_path=str(prepared or source),
            settings=settings,
            codec=codec,
            container=container,
            quality=quality,
            filename_prefix=prefix,
            output_directory=str(output_dir),
            max_frames=max_frames,
            copy_audio=copy_audio,
            verify_neural_rendering=verify,
        )
    finally:
        if prepared is not None:
            prepared.unlink(missing_ok=True)
    rendered, frames = result.args
    return Path(rendered), frames


def _finish_hdr(rendered: Path, settings, hdr: HDROptions, target: Path, sdr: Path) -> None:
    """Write the HDR version of ``rendered`` to ``target``; keep the SDR file as ``sdr`` or drop it."""
    apply_hdr(rendered, settings, hdr, target)
    if not hdr.keep_sdr:
        rendered.unlink(missing_ok=True)
    elif sdr != rendered:
        os.replace(rendered, sdr)


def _output_inputs() -> list:
    """Encoder widgets shared by both render nodes."""
    return [
        io.Combo.Input("codec", display_name="Codec", options=["HEVC", "H.264", "AV1", "ProRes Proxy"], default="HEVC",
                       tooltip="HEVC is the right choice above 4096 px wide (H.264 NVENC stops there)."),
        io.Combo.Input("container", display_name="Container", options=list(_CONTAINER_EXTENSIONS), default="MP4",
                       tooltip="MP4 previews in the browser. MKV keeps subtitles when STEP 2 is off."),
        io.Combo.Input("quality", display_name="Quality", options=["Best", "Good", "Auto", "Max"], default="Best",
                       tooltip="Best measured highest. 'Max' measured WORSE than Good on NVENC."),
        io.Boolean.Input("copy_audio", display_name="Copy audio", default=True),
        io.Float.Input("preview_seconds", default=0.0, min=0.0, max=600.0, step=0.5, advanced=True,
                       tooltip="Set by the Control Panel's Quick preview toggle. 0 renders the whole clip."),
        io.Boolean.Input("verify_neural_rendering", default=True, advanced=True),
    ]


class DLSS5VideoToPath(io.ComfyNode):
    """Expose the file behind a VIDEO input as an absolute path."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DLSS5VideoToPath",
            display_name="Video To File Path",
            category="image/upscaling",
            description=(
                "Returns the absolute path of the file behind a VIDEO input, e.g. from the core "
                "Load Video upload node. The video is not decoded."
            ),
            inputs=[
                io.Video.Input("video"),
                io.Boolean.Input(
                    "batch_mode",
                    default=False,
                    optional=True,
                    tooltip="When on, everything downstream is blocked.",
                ),
            ],
            outputs=[io.String.Output("video_path")],
        )

    @classmethod
    def execute(cls, video, batch_mode: bool = False) -> io.NodeOutput:
        if batch_mode:
            return io.NodeOutput(ExecutionBlocker(None))
        source = video.get_stream_source()
        if not isinstance(source, str):
            raise ValueError(
                "This VIDEO lives in memory, not in a file. Connect a Load Video node "
                "(an uploaded file) instead of a generated video."
            )
        return io.NodeOutput(str(Path(source).resolve()))


class DLSS5PreviewVideoPath(io.ComfyNode):
    """Show a finished video file on the canvas."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DLSS5PreviewVideoPath",
            display_name="Preview Video File",
            category="image/upscaling",
            description=(
                "Previews a video file by path. Only files inside the ComfyUI output folder can "
                "be shown in the browser; other paths are listed as text."
            ),
            is_output_node=True,
            inputs=[io.String.Input("video_path", force_input=True)],
            outputs=[],
        )

    @classmethod
    def execute(cls, video_path: str) -> io.NodeOutput:
        return io.NodeOutput(ui=_preview_ui(Path(video_path)))


class DLSS5ControlPanel(io.ComfyNode):
    """All on/off switches in one place. It has no wires: web/dlss5_panel.js applies the toggles."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DLSS5ControlPanel",
            display_name="DLSS5 Control Panel",
            category="image/upscaling",
            description=(
                "Turns features on and off for the whole workflow. Batch folder swaps the single-video "
                "input/render pair for the batch pair; switching a feature off greys out its settings."
            ),
            inputs=[
                io.Boolean.Input("batch_folder", display_name="Mode", default=False, label_on="Batch folder", label_off="Single video",
                                 tooltip="Render every video in a folder instead of one uploaded video."),
                io.Boolean.Input("rtx_deblur", display_name="RTX deblur", label_on="On", label_off="Off", default=True,
                                 tooltip="STEP 2a: RTX Video Super Resolution deblur before DLSS5 (recommended)."),
                io.Boolean.Input("fps_60", display_name="60 FPS", label_on="On", label_off="Off", default=False,
                                 tooltip="STEP 2b: RIFE up to 60 fps. Doubles+ the render time. Turns 30 FPS off."),
                io.Boolean.Input("fps_30", display_name="30 FPS", label_on="On", label_off="Off", default=False,
                                 tooltip="STEP 2b: RIFE up to 30 fps (24/25 fps clips become 30). Turns 60 FPS off."),
                io.Boolean.Input("hdr", display_name="HDR output", label_on="On", label_off="Off", default=False, tooltip="STEP 4: RTX Video HDR output (10-bit HLG/PQ)."),
                io.Boolean.Input("quick_preview", display_name="Quick preview (5 s)", label_on="On", label_off="Off", default=False,
                                 tooltip="Render only the first 5 seconds to check the look."),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        return io.NodeOutput()


class DLSS5PreprocessSettings(io.ComfyNode):
    """STEP 2: RTX clean pass, RIFE frame rate and RTX pre-upscale, shared by single and batch mode."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DLSS5PreprocessSettings",
            display_name="RTX Clean + 60 FPS Settings",
            category="image/upscaling",
            description=(
                "Runs before DLSS5, in this order: RTX VSR clean pass at source resolution -> RIFE to "
                "target_fps -> optional RTX VSR pre-upscale. On/off lives on the Control Panel."
            ),
            inputs=[
                io.Combo.Input(
                    "rtx_clean", display_name="Clean mode",
                    options=_RTX_CLEAN_MODES,
                    default="DEBLUR_ULTRA",
                    tooltip=(
                        "DEBLUR_ULTRA measured best: it gives DLSS5 sharper material to reconstruct. "
                        "DENOISE_* measured worse in every test; only use it on very noisy sources."
                    ),
                ),
                io.Float.Input("rtx_clean_strength", display_name="Clean strength", default=0.5, min=0.05, max=1.0, step=0.05,
                               tooltip="0.5 measured best; 1.0 adds detail but looks painted on AI footage."),
                io.Float.Input("target_fps", display_name="Target FPS", default=60.0, min=1.0, max=240.0, step=0.001,
                               tooltip="Set by the Control Panel's 60 FPS / 30 FPS toggles."),
                io.Combo.Input(
                    "rtx_pre_upscale", display_name="RTX pre-upscale",
                    options=list(UPSCALE_FACTORS),
                    default="off",
                    tooltip=(
                        "Optional RTX resize before DLSS5. Final size = this x the DLSS5 mode. For 3x total, "
                        "1.5x here + DLSS5 2x measured better than DLSS5 3x."
                    ),
                ),
                io.Combo.Input("rtx_pre_upscale_quality", display_name="Pre-upscale model", options=UPSCALE_QUALITIES, default="HIGHBITRATE_ULTRA", advanced=True),
                io.Float.Input(
                    "scene_cut_threshold", display_name="Cut detection", default=0.30, min=0.05, max=1.0, step=0.01, advanced=True,
                    tooltip="Frames further apart than this are a hard cut and are not interpolated (cuts 0.46-0.66, motion <= 0.22).",
                ),
                io.Boolean.Input("clean_enabled", default=True, advanced=True, tooltip="Set by the Control Panel."),
                io.Boolean.Input("interpolate_fps", default=False, advanced=True, tooltip="Set by the Control Panel."),
            ],
            outputs=[PRE_TYPE.Output("preprocess")],
        )

    @classmethod
    def execute(cls, rtx_clean, rtx_clean_strength, target_fps, rtx_pre_upscale, rtx_pre_upscale_quality="HIGHBITRATE_ULTRA",
                scene_cut_threshold=0.30, clean_enabled=True, interpolate_fps=False) -> io.NodeOutput:
        return io.NodeOutput(PreOptions(
            clean_mode=rtx_clean if clean_enabled else "off", clean_strength=rtx_clean_strength,
            upscale=rtx_pre_upscale, upscale_quality=rtx_pre_upscale_quality, interpolate=interpolate_fps,
            target_fps=target_fps, cut_threshold=scene_cut_threshold,
        ))


class DLSS5HDRSettings(io.ComfyNode):
    """STEP 4: RTX Video HDR applied after DLSS5, shared by single and batch mode."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DLSS5HDRSettings",
            display_name="RTX Video HDR Settings",
            category="image/upscaling",
            description=(
                "Converts the finished DLSS5 render to 10-bit HDR with NVIDIA RTX Video HDR. Runs last: "
                "DLSS5 works in 8-bit SDR and would clip HDR values. On/off lives on the Control Panel."
            ),
            inputs=[
                io.Combo.Input(
                    "colorspace", display_name="HDR format",
                    options=["HLG", "PQ"],
                    default="HLG",
                    tooltip="HLG still looks reasonable on SDR screens. PQ (HDR10) is for HDR TVs/monitors.",
                ),
                io.Int.Input("luminance", display_name="Peak brightness (nits)", default=650, min=400, max=2000, tooltip="Target peak brightness in nits."),
                io.Int.Input("contrast", display_name="Contrast", default=100, min=0, max=200),
                io.Int.Input("saturation", display_name="Saturation", default=100, min=0, max=200),
                io.Int.Input("middle_gray", display_name="Middle grey", default=50, min=10, max=100),
                io.Boolean.Input("debanding", display_name="Debanding", default=True),
                io.Boolean.Input("keep_sdr", display_name="Also keep SDR file", default=True, tooltip="Also keep the SDR render next to the HDR one."),
                io.Boolean.Input("enabled", default=False, advanced=True, tooltip="Set by the Control Panel."),
            ],
            outputs=[HDR_TYPE.Output("hdr")],
        )

    @classmethod
    def execute(cls, colorspace, luminance, contrast, saturation, middle_gray, debanding, keep_sdr, enabled=False) -> io.NodeOutput:
        return io.NodeOutput(HDROptions(
            enabled=enabled, colorspace=colorspace, contrast=contrast, saturation=saturation,
            middle_gray=middle_gray, luminance=luminance, debanding=debanding, keep_sdr=keep_sdr,
        ))


class DLSS5BatchFolderInput(io.ComfyNode):
    """STEP 1 (batch): the folder of clips to render."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DLSS5BatchFolderInput",
            display_name="Batch Folder",
            category="image/upscaling",
            description="Folder of source clips for batch mode (not recursive).",
            inputs=[
                io.String.Input(
                    "input_folder", display_name="Input folder",
                    default="DLSS5_batch",
                    tooltip="A relative path is inside the ComfyUI input folder; an absolute path is used as given.",
                ),
            ],
            outputs=[FOLDER_TYPE.Output("folder")],
        )

    @classmethod
    def execute(cls, input_folder: str) -> io.NodeOutput:
        source_dir = _resolve_folder(input_folder, folder_paths.get_input_directory())
        if not source_dir.is_dir():
            raise FileNotFoundError(f"input_folder {source_dir} does not exist.")
        return io.NodeOutput(str(source_dir))


class DLSS5RenderVideo(io.ComfyNode):
    """STEP 5 (single): pre-pass -> DLSS5 -> HDR for one uploaded video, with a preview."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DLSS5RenderVideo",
            display_name="Render Video",
            category="image/upscaling",
            description=(
                "Runs the whole chain on one video - RTX Clean + 60 FPS, DLSS5, HDR - streaming frame by frame, "
                "and shows the result."
            ),
            is_output_node=True,
            not_idempotent=True,
            inputs=[
                io.Video.Input("video"),
                DLSS5_SETTINGS.Input("dlss5"),
                PRE_TYPE.Input("rtx_clean_60fps"),
                HDR_TYPE.Input("hdr"),
                io.String.Input("filename_prefix", display_name="File name", default="DLSS5_upscale",
                                tooltip="A timestamp is appended; files are never overwritten."),
                io.String.Input("output_folder", display_name="Output folder", default="DLSS5",
                                tooltip="Relative = inside the ComfyUI output folder (previews there); absolute = as given."),
                *_output_inputs(),
            ],
            outputs=[io.String.Output("video_path")],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")

    @classmethod
    def execute(cls, video, dlss5, rtx_clean_60fps: PreOptions, hdr: HDROptions, filename_prefix: str, output_folder: str,
                codec: str, container: str, quality: str, copy_audio: bool, preview_seconds: float = 0.0,
                verify_neural_rendering: bool = True) -> io.NodeOutput:
        source = video.get_stream_source()
        if not isinstance(source, str):
            raise ValueError("This VIDEO lives in memory, not in a file. Connect a Load Video node (an uploaded file).")
        rendered, frames = _render(
            Path(source).resolve(), dlss5, rtx_clean_60fps,
            output_dir=_resolve_folder(output_folder or ".", folder_paths.get_output_directory()),
            prefix=filename_prefix, codec=codec, container=container, quality=quality, copy_audio=copy_audio,
            verify=verify_neural_rendering, preview_seconds=preview_seconds,
        )
        final = rendered
        if hdr.enabled:
            final = rendered.with_name(f"{rendered.stem}_HDR{rendered.suffix}")
            _finish_hdr(rendered, dlss5, hdr, final, sdr=rendered)
        logger.info("DLSS5 render: %s (%d frames)", final, frames)
        return io.NodeOutput(str(final), ui=_preview_ui(final))


class DLSS5BatchEnhanceFolder(io.ComfyNode):
    """STEP 5 (batch): the same chain for every clip in a folder, one after another."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DLSS5BatchEnhanceFolder",
            display_name="Render Folder",
            category="image/upscaling",
            description=(
                "Renders every video from the Batch Folder node, one at a time, and saves each as "
                "<original name><suffix> in output_folder."
            ),
            is_output_node=True,
            not_idempotent=True,
            inputs=[
                FOLDER_TYPE.Input("folder"),
                DLSS5_SETTINGS.Input("dlss5"),
                PRE_TYPE.Input("rtx_clean_60fps"),
                HDR_TYPE.Input("hdr"),
                io.String.Input("output_folder", display_name="Output folder", default="DLSS5_batch",
                                tooltip="Relative = inside the ComfyUI output folder; absolute = as given."),
                io.String.Input("suffix", display_name="Name suffix", default="_DLSS5", tooltip="clip.mp4 -> clip_DLSS5.mp4"),
                io.Boolean.Input("skip_existing", display_name="Skip finished clips", default=True,
                                 tooltip="Skip clips that already have a result, so an interrupted batch resumes."),
                *_output_inputs(),
            ],
            outputs=[
                io.String.Output("written_paths", tooltip="One output path per line."),
                io.Int.Output("rendered_count"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        # Always run: clips may have been added since the last run; skip_existing avoids redoing finished ones.
        return float("nan")

    @classmethod
    def execute(cls, folder: str, dlss5, rtx_clean_60fps: PreOptions, hdr: HDROptions, output_folder: str, suffix: str,
                skip_existing: bool, codec: str, container: str, quality: str, copy_audio: bool,
                preview_seconds: float = 0.0, verify_neural_rendering: bool = True) -> io.NodeOutput:
        source_dir = Path(folder)
        target_dir = _resolve_folder(output_folder, folder_paths.get_output_directory())
        target_dir.mkdir(parents=True, exist_ok=True)
        extension = _CONTAINER_EXTENSIONS[container]

        # Results written next to their sources must not be picked up as new sources.
        clips = sorted(
            (p for p in source_dir.iterdir()
             if p.is_file() and p.suffix.lower() in _VIDEO_EXTENSIONS
             and not p.stem.endswith((suffix, f"{suffix}_SDR"))),
            key=lambda p: p.name.lower(),
        )
        if not clips:
            raise FileNotFoundError(f"No video files in {source_dir}.")

        written, lines, failures = [], [], []
        for index, clip in enumerate(clips, 1):
            target = target_dir / f"{clip.stem}{suffix}{extension}"
            if skip_existing and target.exists():
                lines.append(f"skipped (exists): {target.name}")
                logger.info("DLSS5 batch %d/%d: %s exists, skipping", index, len(clips), target.name)
                continue
            logger.info("DLSS5 batch %d/%d: %s", index, len(clips), clip.name)
            try:
                rendered, frames = _render(
                    clip, dlss5, rtx_clean_60fps, output_dir=target_dir, prefix=_PARTIAL_PREFIX, codec=codec,
                    container=container, quality=quality, copy_audio=copy_audio, verify=verify_neural_rendering,
                    preview_seconds=preview_seconds,
                )
                if hdr.enabled:
                    _finish_hdr(rendered, dlss5, hdr, target, sdr=target.with_name(f"{target.stem}_SDR{target.suffix}"))
                else:
                    os.replace(rendered, target)
            except InterruptProcessingException:
                raise
            except Exception as exc:
                logger.error("DLSS5 batch: %s failed: %s", clip.name, exc)
                failures.append(clip.name)
                lines.append(f"FAILED: {clip.name}: {exc}")
                continue
            written.append(str(target))
            lines.append(f"{target.name} ({frames} frames{', HDR ' + hdr.colorspace if hdr.enabled else ''})")

        summary = (
            f"{len(written)} rendered, {len(clips) - len(written) - len(failures)} skipped, "
            f"{len(failures)} failed -> {target_dir}\n" + "\n".join(lines)
        )
        logger.info("DLSS5 batch: %s", summary.splitlines()[0])
        if failures and not written and len(failures) == len(clips):
            raise RuntimeError(f"Every clip failed.\n{summary}")
        return io.NodeOutput("\n".join(written), len(written), ui=ui.PreviewText(summary))


class Dlss5VideoHelpersExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            DLSS5ControlPanel, DLSS5BatchFolderInput, DLSS5PreprocessSettings, DLSS5HDRSettings,
            DLSS5RenderVideo, DLSS5BatchEnhanceFolder, DLSS5VideoToPath, DLSS5PreviewVideoPath,
        ]


async def comfy_entrypoint() -> Dlss5VideoHelpersExtension:
    return Dlss5VideoHelpersExtension()
