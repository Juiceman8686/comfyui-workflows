"""H3 HQ Ref Pack (from pictures).

Bridge for the H3 Reference Director refine passes (H3RDHQPass2Conditioning) in workflows whose
pass-1 conditioning comes from the core MiniMaxH3ReferenceToVideo node instead of the Reference
Director's own H3SemanticReferenceToVideo. It builds the same `hq_ref_pack` from the pictures that
were wired into MiniMaxH3ReferenceToVideo, in the same order.

If the number of connected pictures does not match the picture references inside the conditioning
(for example in a dialogue / audio-only mode), the pack keeps every picture at its pass-1 size
instead of raising.
"""


class H3RDRefPackFromPictures:
    PICTURES = 9

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING", {"tooltip": "The pass-1 conditioning (output of MiniMaxH3ReferenceToVideo)."}),
                "ref_image_size": (["max", "match"], {
                    "default": "max",
                    "tooltip": "Set this to the same value as ref_image_size on MiniMaxH3ReferenceToVideo. "
                               "Only 'match' pictures are re-encoded at the larger size by the refine passes."}),
            },
            "optional": {f"picture_{i}": ("IMAGE",) for i in range(1, cls.PICTURES + 1)},
        }

    RETURN_TYPES = ("H3_HQ_REFPACK", "STRING")
    RETURN_NAMES = ("hq_ref_pack", "info")
    FUNCTION = "build"
    CATEGORY = "MiniMax H3 / Workflow UI"
    DESCRIPTION = ("Builds the hq_ref_pack for H3 HQ Pass-2 Conditioning from the pictures wired into "
                   "MiniMaxH3ReferenceToVideo. Connect picture_1..9 exactly as they are connected there.")

    def build(self, positive, ref_image_size="max", **pictures):
        images = [pictures.get(f"picture_{i}") for i in range(1, self.PICTURES + 1)]
        images = [img for img in images if img is not None]

        n_blocks = 0
        for entry in positive:
            if isinstance(entry, (list, tuple)) and len(entry) > 1 and isinstance(entry[1], dict):
                blocks = entry[1].get("minimax_refs") or []
                n_blocks = max(n_blocks, sum(1 for blk in blocks if blk.get("kind") == "image"))

        if len(images) == n_blocks:
            packed = [{"logical": i, "role": "", "image": img[:1].detach().cpu()}
                      for i, img in enumerate(images, start=1)]
            info = f"{n_blocks} picture(s) packed, ref_image_size '{ref_image_size}'."
        else:
            packed = [{"logical": i, "role": "", "image": None} for i in range(1, n_blocks + 1)]
            info = (f"{len(images)} picture(s) connected but the conditioning holds {n_blocks} picture "
                    "reference(s); pictures stay at their pass-1 size.")
        print("[H3RDRefPackFromPictures] " + info, flush=True)
        pack = {"version": 1, "width": 0, "height": 0, "ref_image_size": str(ref_image_size), "pictures": packed}
        return pack, info


NODE_CLASS_MAPPINGS = {"H3RDRefPackFromPictures": H3RDRefPackFromPictures}
NODE_DISPLAY_NAME_MAPPINGS = {"H3RDRefPackFromPictures": "H3 HQ Ref Pack (from pictures)"}
