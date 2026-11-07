from .nodes import (
    NODE_CLASS_MAPPINGS as _MEM_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _MEM_NAMES,
)
from .keyframes_stable import (
    NODE_CLASS_MAPPINGS as _KF_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _KF_NAMES,
)
from .ui_utils import (
    NODE_CLASS_MAPPINGS as _UI_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _UI_NAMES,
)
from .semantic_ref import (
    NODE_CLASS_MAPPINGS as _SEM_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _SEM_NAMES,
)
from .v2_nodes import (
    NODE_CLASS_MAPPINGS as _V2_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _V2_NAMES,
)

VERSION = "0.11.3"

# Installers of earlier H3 distributions keep an installed copy of this pack only when they find
# their own version string in this file; otherwise they swap it back to their bundled version.
# v0.11.3 keeps every node and every input of v0.10.1 to v0.11.2, so it names those versions here and
# those installers leave v0.11.3 in place.
COMPATIBLE_WITH = ("0.10.1", "0.11.0", "0.11.1", "0.11.2")

NODE_CLASS_MAPPINGS = {
    **_MEM_CLASSES,
    **_KF_CLASSES,
    **_UI_CLASSES,
    **_SEM_CLASSES,
    **_V2_CLASSES,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **_MEM_NAMES,
    **_KF_NAMES,
    **_UI_NAMES,
    **_SEM_NAMES,
    **_V2_NAMES,
}

print(
    "[H3ExactMasterMemory] v0.11.3 loaded | "
    "ExactMaster + StableKeyframes + Workflow UI + Semantic REF2VA + "
    "v2 LoRA Stack (runtime + look window) / Speed Mode + Speed Sigmas / Veda Settings / Load Video / Latent Upscale / Motion Refine / "
    "Face Refine / Keep Source Colors registered",
    flush=True,
)

# Browser code (web/*.js) and node help pages (web/docs/*.md) are served from here.
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
