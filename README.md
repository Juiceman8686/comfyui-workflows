# ComfyUI Workflows

Workflows I use and may want to share. The workflows have had their input/output media stripped (no pictures, videos, audio or output previews) — load your own files into the loader nodes after opening. The only pictures in the repo are the example images in the HunyuanImage README.

| Workflow | What it is |
|---|---|
| [`HunyuanImage/`](HunyuanImage) | HunyuanImage 3.0 all-in-one workflow: text to image, image editing, multi-image fusion and prompt rewriting from one control panel. Has its own README with models and usage. |
| [`Minimax/Juice_Frankenstein_V7.6.json`](Minimax/Juice_Frankenstein_V7.6.json) | MiniMax H3 reference-to-video workflow with refine passes (Motion Refine, Latent Upscale 1.25x, Face Refine), dialogue/voice-cloning mode and seamless continuation. Based on the original workflow by [roycho87](https://github.com/roycho87), heavily modified. |
| [`Upscale/dlss5_video_upscale.json`](Upscale/dlss5_video_upscale.json) | Video upscale: RTX deblur + RIFE interpolation -> DLSS5 -> RTX HDR. |

## Setup
1. Copy the folders from [`custom_nodes/`](custom_nodes) into your `ComfyUI/custom_nodes/` folder (these have no public home, so they are bundled here — see below), then restart ComfyUI.
2. Install the public packs listed below (ComfyUI-Manager -> "Install Missing Custom Nodes" finds most of them).
3. Drag the `.json` onto the ComfyUI canvas, re-select your models and load your own media in the loader nodes.

## Bundled custom nodes (`custom_nodes/`)
| Folder | Used by | Notes |
|---|---|---|
| [`ComfyUI-H3-ExactMasterMemory`](custom_nodes/ComfyUI-H3-ExactMasterMemory) | V7.6 | H3 Reference Director nodes (`H3RD*`, `H3TrimAVToFrames`). MIT licensed, license file included. |
| [`ComfyUI-H3RD-RefPackBridge`](custom_nodes/ComfyUI-H3RD-RefPackBridge) | V7.6 | `H3RDRefPackFromPictures`: builds the HQ reference pack for the refine passes. |
| [`ComfyUI-DLSS5-VideoHelpers`](custom_nodes/ComfyUI-DLSS5-VideoHelpers) | DLSS5 | Control panel, input and render nodes around the DLSS5 enhancer. Needs an NVIDIA RTX GPU. |

## Public custom nodes to install
**`Juice_Frankenstein_V7.6.json`**
- [ComfyMath](https://github.com/evanspearman/ComfyMath)
- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)
- [ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use)
- [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)
- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
- [ComfyUI-Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation)
- [ComfyUI_essentials](https://github.com/cubiq/ComfyUI_essentials)
- [ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts)
- [rgthree-comfy](https://github.com/rgthree/rgthree-comfy)
- [ComfyUI-basic_data_handling](https://github.com/StableLlama/ComfyUI-basic_data_handling)
- [CRT-Nodes](https://github.com/PGCRT/CRT-Nodes)
- [ComfyUI-Pixaroma](https://gitlab.com/pixaroma/ComfyUI-Pixaroma)
- [ComfyUI-MAINodes](https://github.com/matlowai/ComfyUI-MAINodes)
- [ComfyUI-AnimatedMaskEditor](https://github.com/BISAM20/ComfyUI-AnimatedMaskEditor)
- [MaskVidExperiments](https://github.com/drozbay/MaskVidExperiments)
- [comfyui-video-resolution](https://github.com/hyukudan/comfyui-video-resolution)
- [Comfyui-PainterAudioLength](https://github.com/princepainter/Comfyui-PainterAudioLength)
- [ComfyUI-Fantastic-MiniMaxH3-PromptBuilder](https://github.com/Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder)
- [ComfyUI-H3-Prompt-IDE](https://github.com/ethanfel/ComfyUI-H3-Prompt-IDE)
- [ComfyUI-H3-FaceRefine](https://github.com/Carasibana/ComfyUI-H3-FaceRefine)
- [H3-Optimizations](https://github.com/Zironic/H3-Optimizations)
- [Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)
- [Veda-on-ComfyUI](https://github.com/veda-sparse/Veda-on-ComfyUI) (only for the Veda sparse-attention option)

**`dlss5_video_upscale.json`**
- [ComfyUI-DLSS5-Enhancer](https://github.com/Blueforcer/ComfyUI-DLSS5-Enhancer)
- ComfyUI core RTX nodes (current ComfyUI)

## Notes
- Model file names reflect my local setup; pick your own equivalents in the model loaders.
- Model download links for the MiniMax H3 models are in the "Model Links" note inside V7.6.
