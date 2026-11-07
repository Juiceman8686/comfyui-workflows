from __future__ import annotations

class H3ResolutionPreset:
    PRESETS = {
        "Fast Landscape (864x480)": (864, 480),
        "Fast Portrait (480x864)": (480, 864),
        "Test Landscape (960x544)": (960, 544),
        "Test Portrait (544x960)": (544, 960),
        "H3 Native Landscape (1344x768)": (1344, 768),
        "H3 Native Portrait (768x1344)": (768, 1344),
        "H3 4:3 Landscape (1024x768)": (1024, 768),
        "H3 3:4 Portrait (768x1024)": (768, 1024),
        "H3 Square (768x768)": (768, 768),
        "H3 Cinematic 21:9 (1536x672)": (1536, 672),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (list(cls.PRESETS.keys()), {
                    "default": "H3 Native Landscape (1344x768)"
                }),
            }
        }

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "info")
    FUNCTION = "pick"
    CATEGORY = "MiniMax H3 / Workflow UI"
    DESCRIPTION = "Friendly MiniMax H3 resolution presets for clean beginner workflows."

    def pick(self, preset):
        w, h = self.PRESETS[preset]
        info = f"{preset} | {w}x{h}"
        print(f"[H3ResolutionPreset] {info}", flush=True)
        return int(w), int(h), info


class H3DurationPlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seconds": ("INT", {
                    "default": 6, "min": 2, "max": 15, "step": 1,
                    "tooltip": "Requested final delivered video duration in seconds."
                }),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0
                }),
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("gen_length", "output_frames", "actual_seconds", "info")
    FUNCTION = "plan"
    CATEGORY = "MiniMax H3 / Workflow UI"
    DESCRIPTION = (
        "Converts simple seconds to a valid MiniMax H3 generation length. "
        "Outputs both the H3-safe generation frame count and the exact final "
        "delivery frame count for post-trim."
    )

    def plan(self, seconds, fps):
        target_frames = int(round(float(seconds) * float(fps)))
        # MiniMax H3 frame grid used in our validated workflows: 17k + 5.
        k = max(0, int((target_frames - 5 + 16) // 17))
        gen_length = 5 + 17 * k
        if gen_length < target_frames:
            gen_length += 17
        # Keep open-source H3 generation inside the documented ~124f+ trained range,
        # then trim decoded A/V to the requested exact delivery duration.
        gen_length = max(124, gen_length)
        actual_seconds = target_frames / float(fps)
        info = (
            f"deliver={seconds}s ({target_frames}f @ {fps:g}fps) | "
            f"generate={gen_length}f ({gen_length/float(fps):.2f}s) | "
            f"trim_to={target_frames}f"
        )
        print(f"[H3DurationPlanner] {info}", flush=True)
        return int(gen_length), int(target_frames), float(actual_seconds), info


class H3TrimAVToFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "output_frames": ("INT", {"default": 144, "min": 1, "max": 5000, "step": 1}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "info")
    FUNCTION = "trim"
    CATEGORY = "MiniMax H3 / Workflow UI"
    DESCRIPTION = "Trims decoded H3 video frames and decoded audio to the exact target duration."

    def trim(self, images, audio, output_frames, fps):
        import torch

        output_frames = int(output_frames)
        fps = float(fps)
        trimmed_images = images[:output_frames]

        # AUDIO inputs are mapping-like objects. Core/decoded H3 audio is normally a
        # plain dict, while lazy media loaders (for example VideoHelperSuite video
        # audio) may provide a read-only LazyAudioMap. Never mutate or deepcopy the
        # input mapping; materialize a fresh standard ComfyUI AUDIO dict instead.
        sample_rate = int(audio["sample_rate"])
        target_samples = int(round((output_frames / fps) * sample_rate))
        waveform = audio["waveform"]

        if waveform.ndim == 3:
            trimmed_waveform = waveform[..., :target_samples]
        elif waveform.ndim == 2:
            trimmed_waveform = waveform[:, :target_samples]
        else:
            trimmed_waveform = waveform[:target_samples]

        out_audio = {
            "waveform": trimmed_waveform.contiguous(),
            "sample_rate": sample_rate,
        }

        info = (
            f"trimmed_to={output_frames}f @ {fps:g}fps | "
            f"audio_samples={target_samples}"
        )
        print(f"[H3TrimAVToFrames] {info}", flush=True)
        return trimmed_images, out_audio, info


NODE_CLASS_MAPPINGS = {
    "H3ResolutionPreset": H3ResolutionPreset,
    "H3DurationPlanner": H3DurationPlanner,
    "H3TrimAVToFrames": H3TrimAVToFrames,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ResolutionPreset": "H3 Resolution Preset",
    "H3DurationPlanner": "H3 Duration Planner",
    "H3TrimAVToFrames": "H3 Trim A/V To Frames",
}
