"""Stable arbitrary-position keyframes for MiniMax H3.

Uses the already-installed H3 Motion Context PackedLayout wrapper instead of
rewriting ComfyUI source text. This is robust against source formatting changes
that break the older h3_interior_patch.py literal replacement.
"""
from __future__ import annotations

MAX_SLOTS = 6
MC_KEY = "motion_context_index"
MC_PATCH_MARKER = "_h3_motion_context_layout_patch"


def _parse_positions(text, frame_count):
    raw = (text or "").replace(";", ",")
    toks = [t.strip() for t in raw.split(",") if t.strip()]
    if not toks:
        raise ValueError("positions is empty")

    if "%" not in raw and "-" not in raw:
        vals = []
        for t in toks:
            try:
                vals.append(float(t))
            except ValueError:
                vals = None
                break
        if vals and any(v != int(v) and 0 <= v <= 1 for v in vals):
            return [max(0, min(frame_count - 1,
                    int(round(v * (frame_count - 1))))) for v in vals]

    def parse_value(tok):
        tok = tok.strip()
        if tok.startswith("-"):
            raise ValueError(f"position {tok} is negative")
        is_pct = tok.endswith("%")
        body = tok[:-1] if is_pct else tok
        try:
            value = float(body)
        except ValueError:
            raise ValueError(f"'{body}' in positions is not a number")
        if value < 0:
            raise ValueError(f"position {tok} is negative")
        return value, is_pct

    def to_index(value, is_pct):
        idx = int(round((value / 100.0) * (frame_count - 1))) if is_pct else int(round(value))
        return max(0, min(frame_count - 1, idx))

    out = []
    for tok in toks:
        if tok.startswith("-"):
            raise ValueError(f"position {tok} is negative")
        if "-" in tok:
            parts = tok.split("-")
            if len(parts) != 2:
                raise ValueError(f"invalid range '{tok}'")
            lv, lp = parse_value(parts[0])
            rv, rp = parse_value(parts[1])
            if lp != rp:
                raise ValueError(f"range '{tok}' mixes percentages and frame indices")
            a, b = to_index(lv, lp), to_index(rv, rp)
            step = 1 if a <= b else -1
            out.extend(range(a, b + step, step))
        else:
            v, p = parse_value(tok)
            out.append(to_index(v, p))
    return out


def _motion_context_patch_active():
    """Probe behavior instead of trusting a patch-marker attribute.

    Different Motion Context revisions/forks can support motion_context_index
    correctly while exposing different wrapper metadata. We only need to know
    whether PackedLayout actually places a frame-0 carrier row at the requested
    interior frame when motion_context_index=N is present.
    """
    try:
        import comfy.ldm.minimax.model as mm

        text_len = 7
        latent_t = 7
        latent_h = 4
        latent_w = 4
        audio_t = 8
        frame_count = sum(
            mm.FRAME_PER_TOKEN[k % len(mm.FRAME_PER_TOKEN)]
            for k in range(latent_t)
        )
        probe_frame = min(3, frame_count - 2)
        if probe_frame <= 0:
            return False

        layout = mm.PackedLayout(
            text_len, latent_t, latent_h, latent_w, audio_t,
            keyframes=[{
                "resolved_frame_index": 0,
                MC_KEY: int(probe_frame),
            }],
            refs=None,
            frame_count=frame_count,
        )

        cond_spans = [(a, b) for a, b, kind in layout.segments if kind == "cond"]
        if len(cond_spans) != 1:
            return False
        a, b = cond_spans[0]
        if b <= a:
            return False

        got = float(layout.position_ids[a, 0])
        expected = float(text_len) + float(mm.FRAME_RESCALE) * float(probe_frame)
        return abs(got - expected) <= 1e-9
    except Exception as exc:
        print(
            f"[H3StableKeyframes] Motion Context capability probe failed: {exc}",
            flush=True,
        )
        return False

class H3StableKeyframes:
    @classmethod
    def INPUT_TYPES(cls):
        opt = {f"image_{i}": ("IMAGE",) for i in range(1, MAX_SLOTS + 1)}
        opt["images_batch"] = ("IMAGE", {"tooltip": "Optional IMAGE batch appended after image_N slots."})
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "width": ("INT", {"default": 960, "min": 32, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 544, "min": 32, "max": 4096, "step": 32}),
                "length": ("INT", {"default": 243, "min": 5, "max": 3600, "step": 17}),
                "positions": ("STRING", {"default": "0%, 50%, 100%"}),
            },
            "optional": opt,
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("positive", "latent", "info")
    FUNCTION = "build"
    CATEGORY = "conditioning/video_models"
    DESCRIPTION = "MiniMax H3 arbitrary visual keyframes using Motion Context's robust positional wrapper."

    def build(self, clip, vae, prompt, width, height, length, positions, **kwargs):
        import torch
        import node_helpers
        from comfy_extras import nodes_minimax_h3 as mmh3

        latent, frame_count = mmh3._empty_av_latent(width, height, length)
        imgs = [kwargs.get(f"image_{i}") for i in range(1, MAX_SLOTS + 1)]
        imgs = [im for im in imgs if im is not None]
        batch = kwargs.get("images_batch")
        if batch is not None and int(batch.shape[0]) > 0:
            imgs.extend(list(torch.chunk(batch, chunks=int(batch.shape[0]), dim=0)))
        if not imgs:
            raise ValueError("H3 Stable Keyframes: connect at least one image.")

        idxs = _parse_positions(positions, frame_count)
        if len(idxs) != len(imgs):
            raise ValueError(
                f"H3 Stable Keyframes: {len(imgs)} image(s) connected but "
                f"{len(idxs)} position(s) supplied ('{positions}').")

        order = sorted(range(len(idxs)), key=lambda i: idxs[i])
        idxs = [idxs[i] for i in order]
        imgs = [imgs[i] for i in order]
        for a, b in zip(idxs, idxs[1:]):
            if a == b:
                raise ValueError(f"H3 Stable Keyframes: two anchors resolve to frame {a}.")

        interior = [i for i in idxs if i not in (0, frame_count - 1)]
        patch_active = _motion_context_patch_active()
        if interior and not patch_active:
            raise RuntimeError(
                "H3 Stable Keyframes: interior anchors require the current "
                "ComfyUI-H3-Motion-Context positional patch. Install/update "
                "ComfyUI-H3-Motion-Context, restart ComfyUI, and confirm the "
                "console says 'h3_motion_context: interior keyframe anchors enabled'.")

        keyframes, enc_images, notes = [], [], []
        for img, idx in zip(imgs, idxs):
            mode = "disabled" if idx == 0 else "center"
            resized = mmh3._resize(img[:1], width, height, mode)
            enc_images.append(resized)
            if patch_active:
                kf = {"resolved_frame_index": 0, MC_KEY: int(idx), "image": resized}
            else:
                kf = {"resolved_frame_index": int(idx), "image": resized}
            keyframes.append(kf)
            notes.append(f"{idx}({idx/24.0:.2f}s)")

        tokens = clip.tokenize(prompt, images=enc_images)
        cond = clip.encode_from_tokens_scheduled(tokens)
        for kf in keyframes:
            kf["latent"] = vae.encode(kf.pop("image"))

        cond = node_helpers.conditioning_set_values(cond, {
            "minimax_keyframes": keyframes,
            "minimax_frame_count": frame_count,
        })

        import comfy.model_management as mm
        try:
            clip.patcher.model.to(mm.text_encoder_offload_device())
        except Exception as e:
            print(f"[H3StableKeyframes] TE offload skipped: {e}", flush=True)
        try:
            dev = mm.get_torch_device()
            mm.free_memory(mm.get_total_memory(dev) * 0.9, dev)
            mm.soft_empty_cache()
            free = mm.get_free_memory(dev) / (1024 ** 3)
            print(f"[H3StableKeyframes] TE evicted; {free:.1f} GB free", flush=True)
        except Exception as e:
            print(f"[H3StableKeyframes] VRAM purge skipped: {e}", flush=True)

        method = "MotionContext positional wrapper" if patch_active else "stock endpoints"
        info = (f"{frame_count} frames ({frame_count/24.0:.2f}s) | "
                f"{len(keyframes)} keyframe(s) at {', '.join(notes)} | {method}")
        print("[H3StableKeyframes] " + info, flush=True)
        return cond, latent, info


NODE_CLASS_MAPPINGS = {"H3StableKeyframes": H3StableKeyframes}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3StableKeyframes": "H3 Keyframes - Stable Interior (MotionContext Patch)"
}
