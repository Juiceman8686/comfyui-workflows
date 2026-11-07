from __future__ import annotations

import math
import re
import node_helpers


def _resample_audio(waveform, sr, target_sr):
    """Resample without a hard torchaudio dependency (v0.5.0).

    torchaudio is tried first so existing installs keep byte-identical behaviour with
    v0.4.x. Newer ComfyUI builds no longer ship torchaudio, so ComfyUI's own
    comfy.audio.resample is the fallback.
    """
    if int(sr) == int(target_sr):
        return waveform
    try:
        import torchaudio
        return torchaudio.functional.resample(waveform, sr, target_sr)
    except Exception:
        pass
    try:
        import comfy.audio
        return comfy.audio.resample(waveform, int(sr), int(target_sr))
    except Exception as exc:
        raise RuntimeError(
            "H3 Semantic Reference Director: reference audio is %d Hz but the H3 audio VAE "
            "needs %d Hz, and neither torchaudio nor comfy.audio.resample is available. "
            "Update ComfyUI or install torchaudio, or supply %d Hz audio." % (sr, target_sr, target_sr)
        ) from exc


class H3SemanticReferenceToVideo:
    """
    Fixed-role semantic wrapper around MiniMax H3 REF2VA.

    Stock H3 compacts only ACTIVE refs before assigning <Picture i>/<Video k>/<Audio j>.
    This node lets a workflow/user keep stable logical roles and remaps those logical tags
    to the active compact H3 ordinals immediately before tokenization.
    """

    CANVAS_MULTIPLE = 32
    REF_IMAGE_SHORT_EDGE = 2048
    FPS = 24

    IMAGE_ROLES = [
        ("face", 1, "FACE identity"),
        ("character_sheet", 2, "CHARACTER DATA SHEET"),
        ("wardrobe", 3, "WARDROBE only"),
        ("product", 4, "PRODUCT / hero object"),
        ("style", 5, "VISUAL STYLE"),
        ("lighting", 6, "GLOBAL LIGHTING / grade"),
        ("background", 7, "BACKGROUND / environment"),
        ("extra_1", 8, "OPTIONAL EXTRA 1"),
        ("extra_2", 9, "OPTIONAL EXTRA 2"),
    ]
    VIDEO_ROLES = [
        ("motion_video", 1, "MOTION guide"),
        ("camera_video", 2, "CAMERA guide"),
        ("source_video", 3, "SOURCE / edit video"),
    ]
    AUDIO_ROLES = [
        ("audio_primary", 1, "PRIMARY audio"),
        ("audio_2", 2, "OPTIONAL audio 2"),
    ]
    SOURCE_VIDEO_AUDIO_LOGICAL = 3

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 1344, "min": 32, "max": 16384, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": 16384, "step": 32}),
                "length": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17}),
                "ref_image_size": (["match", "max"], {"default": "match"}),
                "wardrobe_head_crop_pct": ("FLOAT", {
                    "default": 18.0, "min": 0.0, "max": 40.0, "step": 1.0,
                    "tooltip": (
                        "For wardrobe photos containing another person, crop this percentage "
                        "from the top before H3 sees the wardrobe reference. Set 0 for flat-lay."
                    ),
                }),
                "strict_missing_tags": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "face": ("IMAGE",),
                "character_sheet": ("IMAGE",),
                "wardrobe": ("IMAGE",),
                "product": ("IMAGE",),
                "style": ("IMAGE",),
                "lighting": ("IMAGE",),
                "background": ("IMAGE",),
                "extra_1": ("IMAGE",),
                "extra_2": ("IMAGE",),

                "motion_video": ("IMAGE",),
                "camera_video": ("IMAGE",),
                "source_video": ("IMAGE",),

                "audio_primary": ("AUDIO",),
                "audio_2": ("AUDIO",),
                "source_video_audio": ("AUDIO",),

                # v0.5.0: optional text placed in front of the prompt. Wire the
                # trigger_prefix output of "H3 LoRA Stack" here so LoRA trigger
                # words are added only while that LoRA is switched on.
                "prompt_prefix": ("STRING", {"forceInput": True}),
            },
        }

    # v0.5.0 adds a 5th output. Outputs 1-4 are unchanged, so v1 workflows still load.
    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING", "STRING", "H3_HQ_REFPACK")
    RETURN_NAMES = ("positive", "latent", "resolved_prompt", "reference_map", "hq_ref_pack")
    FUNCTION = "encode"
    CATEGORY = "MiniMax H3 / Workflow UI"

    @staticmethod
    def _h3():
        from comfy_extras import nodes_minimax_h3
        return nodes_minimax_h3

    @staticmethod
    def _encode_ref_audio(audio_vae, audio):
        waveform = audio["waveform"]
        sr = audio["sample_rate"]
        vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
        if sr != vae_sr:
            waveform = _resample_audio(waveform, sr, vae_sr)
        z = audio_vae.encode(waveform[:1].movedim(1, -1))
        return z, z.shape[-1]

    @classmethod
    def _logical_maps_from_inputs(cls, kwargs):
        """
        Build logical->actual compact ordinals from input PRESENCE only.
        This is intentionally lightweight so prompt validation can happen
        before any expensive VAE encoding.
        """
        picture_map = {}
        video_map = {}
        audio_map = {}

        actual = 1
        for key, logical, _role in cls.IMAGE_ROLES:
            if kwargs.get(key) is not None:
                picture_map[logical] = actual
                actual += 1

        actual = 1
        for key, logical, _role in cls.VIDEO_ROLES:
            if kwargs.get(key) is not None:
                video_map[logical] = actual
                actual += 1

        external_audio_active = any(kwargs.get(key) is not None for key, _logical, _role in cls.AUDIO_ROLES)
        source_soundtrack_active = (
            kwargs.get("source_video") is not None
            and kwargs.get("source_video_audio") is not None
            and not external_audio_active
        )

        # Official H3 presentation order puts a reference video's paired soundtrack
        # immediately before that video, while standalone audio follows the videos.
        # Workflow 01 deliberately makes these mutually exclusive: Audio 1/2 override
        # Video 3's source soundtrack. This keeps logical Audio 3 permanently reserved
        # for the Video 3 soundtrack without creating ordinal conflicts.
        actual = 1
        if source_soundtrack_active:
            audio_map[cls.SOURCE_VIDEO_AUDIO_LOGICAL] = actual
            actual += 1
        else:
            for key, logical, _role in cls.AUDIO_ROLES:
                if kwargs.get(key) is not None:
                    audio_map[logical] = actual
                    actual += 1

        return picture_map, video_map, audio_map

    @staticmethod
    def _rewrite_family(prompt, family, mapping, strict):
        maximum = {"Picture": 9, "Video": 3, "Audio": 3}[family]
        missing = []

        for logical in range(1, maximum + 1):
            angle = rf"<{family}\s+{logical}>"
            plain = rf"\b{family}\s+{logical}\b"
            if logical not in mapping:
                if re.search(angle, prompt, re.I) or re.search(plain, prompt, re.I):
                    missing.append(f"{family} {logical}")

        if missing and strict:
            raise ValueError(
                "H3 Semantic Reference Director: prompt references media that is not active: "
                + ", ".join(missing)
                + ". Remove inactive reference tags from the creative prompt or enable those media inputs."
            )

        out = prompt
        for logical, actual in mapping.items():
            tmp = f"__H3_{family.upper()}_{logical}__"
            out = re.sub(rf"<{family}\s+{logical}>", tmp, out, flags=re.I)
            out = re.sub(rf"\b{family}\s+{logical}\b", tmp, out, flags=re.I)

        for logical, actual in mapping.items():
            out = out.replace(f"__H3_{family.upper()}_{logical}__", f"<{family} {actual}>")

        return out

    def encode(
        self,
        clip,
        vae,
        audio_vae,
        prompt,
        width,
        height,
        length,
        ref_image_size="match",
        wardrobe_head_crop_pct=18.0,
        strict_missing_tags=True,
        prompt_prefix=None,
        **kwargs,
    ):
        h3 = self._h3()

        if width % self.CANVAS_MULTIPLE or height % self.CANVAS_MULTIPLE:
            raise ValueError(
                f"MiniMax H3 canvas must be divisible by 32; received {width}x{height}."
            )

        # PRE-FLIGHT FIRST: validate/remap the creative prompt before any VAE work.
        # v0.4.0 did this only after encoding all refs, wasting time on a bad tag.
        picture_map, video_map, audio_map = self._logical_maps_from_inputs(kwargs)

        total_preflight = len(picture_map) + len(video_map) + len(audio_map)
        if total_preflight > 12:
            raise ValueError(
                f"{total_preflight} reference files are active. Official H3 mixed-reference guidance "
                "is no more than 12; disable lower-priority optional references."
            )

        resolved = prompt
        resolved = self._rewrite_family(resolved, "Picture", picture_map, bool(strict_missing_tags))
        resolved = self._rewrite_family(resolved, "Video", video_map, bool(strict_missing_tags))
        resolved = self._rewrite_family(resolved, "Audio", audio_map, bool(strict_missing_tags))

        # v0.5.0: LoRA trigger words go in front of the finished prompt. They are
        # added after tag remapping so they can never be mistaken for a reference tag.
        prefix = (prompt_prefix or "").strip()
        if prefix:
            resolved = prefix.rstrip(", ") + ", " + resolved

        latent, frame_count = h3._empty_av_latent(width, height, length)
        ref_items = []
        ref_blocks = []
        mapping_lines = []

        # ----- Images -----
        active_images = []
        for key, logical, role in self.IMAGE_ROLES:
            img = kwargs.get(key)
            if img is not None:
                if key == "wardrobe" and wardrobe_head_crop_pct > 0:
                    hh = img.shape[1]
                    y0 = min(hh - 1, max(0, round(hh * wardrobe_head_crop_pct / 100.0)))
                    img = img[:, y0:, :, :]
                active_images.append((logical, role, img))

        hq_pictures = []
        picture_map = {}
        for actual, (logical, role, img) in enumerate(active_images, start=1):
            picture_map[logical] = actual
            hq_pictures.append({
                "logical": logical,
                "role": role,
                # source picture after the wardrobe crop, kept on CPU at its own size
                "image": img[:1].detach().cpu(),
            })

            ih, iw = img.shape[1], img.shape[2]
            if ref_image_size == "match":
                scale = min(1.0, math.sqrt((width * height) / (iw * ih)))
            else:
                scale = min(1.0, self.REF_IMAGE_SHORT_EDGE / min(iw, ih))

            tw = max(32, round(iw * scale / 32) * 32)
            th = max(32, round(ih * scale / 32) * 32)

            resized = h3._resize(img[:1], tw, th, "disabled")
            z = vae.encode(resized)

            ref_items.append({"type": "image", "data": resized})
            ref_blocks.append({
                "kind": "image",
                "latent_h": th // 16,
                "latent_w": tw // 16,
                "latent": z,
            })
            mapping_lines.append(
                f"logical <Picture {logical}> ({role}) -> active <Picture {actual}>"
            )

        # ----- Videos -----
        external_audio_active = any(kwargs.get(key) is not None for key, _logical, _role in self.AUDIO_ROLES)
        use_source_soundtrack = (
            kwargs.get("source_video") is not None
            and kwargs.get("source_video_audio") is not None
            and not external_audio_active
        )

        active_videos = []
        for key, logical, role in self.VIDEO_ROLES:
            frames = kwargs.get(key)
            if frames is not None:
                active_videos.append((key, logical, role, frames))

        video_map = {}
        for actual, (key, logical, role, video_frames) in enumerate(active_videos, start=1):
            video_map[logical] = actual

            vh, vw = video_frames.shape[1], video_frames.shape[2]
            cw, ch = h3.adapt_canvas(vw, vh)
            if vw * vh < cw * ch:
                cw = max(32, round(vw / 32) * 32)
                ch = max(32, round(vh / 32) * 32)

            frames = h3._resize(video_frames, cw, ch, "disabled")
            if frames.shape[0] > frame_count:
                frames = frames[:frame_count]

            n = frames.shape[0]
            if n < 5:
                raise ValueError("MiniMax H3 reference videos need at least 5 frames.")
            while n % 17 != 5:
                n -= 1
            frames = frames[:n]

            z = vae.encode(frames)
            audio_latent, ref_audio_t = None, 0
            if key == "source_video" and use_source_soundtrack:
                audio_latent, ref_audio_t = self._encode_ref_audio(
                    audio_vae, kwargs.get("source_video_audio")
                )
                # Match ComfyUI core: a paired soundtrack gets its own Audio label
                # immediately before the corresponding Video label in the Qwen context.
                ref_items.append({"type": "audio"})
                mapping_lines.append(
                    "logical <Audio 3> (VIDEO 3 source soundtrack) -> active <Audio 1>"
                )

            sample_idx = list(range(0, frames.shape[0], self.FPS // 2))
            qwen_frames = frames[sample_idx]

            ref_items.append({
                "type": "video",
                "data": qwen_frames,
                "timestamps": [i / 2.0 for i in range(len(sample_idx))],
            })
            ref_blocks.append({
                "kind": "video_audio" if ref_audio_t else "video",
                "latent_t": z.shape[2],
                "latent_h": ch // 16,
                "latent_w": cw // 16,
                "ref_audio_t": ref_audio_t,
                "latent": z,
                "audio_latent": audio_latent,
            })
            mapping_lines.append(
                f"logical <Video {logical}> ({role}) -> active <Video {actual}>"
            )

        # ----- Audio -----
        active_audio = []
        for key, logical, role in self.AUDIO_ROLES:
            audio = kwargs.get(key)
            if audio is not None:
                active_audio.append((logical, role, audio))

        audio_map = {}
        for actual, (logical, role, audio) in enumerate(active_audio, start=1):
            audio_map[logical] = actual
            audio_latent, ref_audio_t = self._encode_ref_audio(audio_vae, audio)

            ref_items.append({"type": "audio"})
            ref_blocks.append({
                "kind": "audio",
                "ref_audio_t": ref_audio_t,
                "audio_latent": audio_latent,
            })
            mapping_lines.append(
                f"logical <Audio {logical}> ({role}) -> active <Audio {actual}>"
            )

        total = (
            len(active_images)
            + len(active_videos)
            + len(active_audio)
            + (1 if use_source_soundtrack else 0)
        )
        if total > 12:
            raise ValueError(
                f"{total} reference files are active. Official H3 mixed-reference guidance "
                "is no more than 12; disable lower-priority optional references."
            )

        # The compact maps used for prompt rewriting were computed before VAE
        # encoding. The encoded active order must match those same maps.
        tokens = clip.tokenize(resolved, minimax_ref_items=ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)

        if ref_blocks:
            cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})

        mapping = "\n".join(mapping_lines) if mapping_lines else "No references active."
        print("[H3SemanticRef] active mapping:\n" + mapping, flush=True)
        print("[H3SemanticRef] resolved prompt:\n" + resolved, flush=True)

        hq_ref_pack = {
            "version": 1,
            "width": int(width),
            "height": int(height),
            "ref_image_size": str(ref_image_size),
            "pictures": hq_pictures,
        }
        return cond, latent, resolved, mapping, hq_ref_pack


class H3SourceEditAudioPriority:
    """
    Workflow 01 audio-output switch dedicated to Video 3 source editing.

    Priority:
      1) If Audio 1 or Audio 2 is active, keep H3's generated/native audio.
      2) Otherwise, if Video 3 provides a soundtrack, pass that ORIGINAL AUDIO
         object through unchanged so the final workflow can trim/remux it.
      3) Otherwise, fall back to H3's generated/native audio.

    This node never resamples or regenerates the selected source soundtrack.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_audio": ("AUDIO",),
            },
            "optional": {
                "source_video_audio": ("AUDIO",),
                "audio_primary": ("AUDIO",),
                "audio_2": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")
    FUNCTION = "route"
    CATEGORY = "MiniMax H3 / Workflow UI"

    def route(
        self,
        generated_audio,
        source_video_audio=None,
        audio_primary=None,
        audio_2=None,
    ):
        external_audio_active = audio_primary is not None or audio_2 is not None

        if source_video_audio is not None and not external_audio_active:
            info = (
                "Video 3 source soundtrack selected unchanged for final output; "
                "Audio 1/2 are inactive."
            )
            print("[H3SourceEditAudioPriority] " + info, flush=True)
            return source_video_audio, info

        if external_audio_active:
            info = (
                "Audio 1/2 override active; using H3 generated/native audio. "
                "Video 3 source soundtrack is not used for final output."
            )
        else:
            info = (
                "No Video 3 source soundtrack available; using H3 generated/native audio."
            )

        print("[H3SourceEditAudioPriority] " + info, flush=True)
        return generated_audio, info


NODE_CLASS_MAPPINGS = {
    "H3SemanticReferenceToVideo": H3SemanticReferenceToVideo,
    "H3SourceEditAudioPriority": H3SourceEditAudioPriority,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3SemanticReferenceToVideo": "H3 Semantic Reference Director — Auto Remap",
    "H3SourceEditAudioPriority": "H3 Video 3 Source Audio Priority",
}
