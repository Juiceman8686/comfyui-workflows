# ComfyUI-H3-ExactMasterMemory

Custom nodes for the MiniMax H3 Reference Director Pro long-form exact-audio path.

## Main node: H3 Exact Master Audio + Memory Sampler

This fills the specific gap in the current H3 ecosystem:

- long-form visual memory / identity continuity;
- fixed REF2VA references across the chain;
- exact prerecorded master audio driving every shot;
- no regenerated/concatenated H3 soundtrack.

It does **not** add a second model or a second diffusion pass.

### Performance design
- master audio resampled once;
- fixed reference images encoded once;
- one audio-VAE target encode per shot;
- generated H3 audio is never decoded;
- final output audio is the original source master timeline;
- text encoder is evicted before sampling when it shares the DiT GPU.

### Recommended Workflow 02 path
REF2VA + Sage + RES multistep + simple + 20 steps.
Use fixed identity/product/style references and memory_frames=2, anchor_frames=1.

### Dependency
When `reference_images` are connected, install the current
`jlucasmcrell/ComfyUI-H3-Multishot`. Its runtime AV-bank patch allows REF2VA
reference rows and chaining keyframes to coexist.

## Utility node: H3 Exact Master Audio Segment
Use this in Motion Context exact-audio graphs. Set overlap_frames to the actual
pinned head size (for example 22) so the H3 generation receives the correct
overlapping slice of the original master soundtrack.

## Validation
- Python source syntax checked.
- Timeline mathematics unit tested.
- v0.11.3 was rendered and approved on a GPU with Reference Director 01 v2.1.3: the
  normal video (close-ups, medium and full-body shots), the upscaled video with and
  without the colour lock, and the face refine. The motion refine and video loader
  nodes were tested without a GPU: try them on a short clip first.

## Research basis
- ComfyUI native MiniMax H3 primitives
- ComfyUI-H3-Multishot (MIT)
- h3-audio-injection (MIT)
- MiniMax-H3-NativeAudio-MusicVideo-Workflow (technique reference)
- H3 Motion Context (GPL-3.0; used as an external workflow dependency, no code copied)

## License
This package source is MIT. Models and dependencies retain their own licenses.

## v0.2.0 - Stable interior H3 keyframes

Adds **H3 Keyframes - Stable Interior (MotionContext Patch)**.

This replaces the brittle exact-source-text rewrite used by the older H3
interior-keyframe patch with the current H3 Motion Context positional-wrapper
ABI. Current `ComfyUI-H3-Motion-Context` is required for interior anchors.
First/last-only anchors still work without it.

## v0.2.2 — duplicate-folder cleanup + behavioral patch probe

- `H3StableKeyframes` no longer requires a particular marker attribute.
  It builds a tiny `PackedLayout` probe and verifies that
  `motion_context_index=N` really moves the cond rows to frame N.
- Backup/duplicate `ComfyUI-H3-ExactMasterMemory*` folders inside
  `custom_nodes` are explicitly warned about. Renaming a backup folder is not
  sufficient because ComfyUI imports every valid custom-node folder there.
- The companion installer moves legacy backup copies outside `custom_nodes`.

## v0.3.0 — Workflow UI utility nodes

Adds three beginner-facing utility nodes used by the upgraded Workflow 01:

- **H3 Resolution Preset**
  - one-click resolution/orientation presets
  - avoids raw width/height editing

- **H3 Duration Planner**
  - user enters simple target seconds
  - converts to a valid H3 generation frame count on the 17k+5 grid
  - also outputs exact final delivery frame count

- **H3 Trim A/V To Frames**
  - trims decoded video frames and decoded audio to the requested final duration

This keeps the frontend much cleaner while preserving the validated H3 backend path.


## v0.4.0 — Semantic reference auto-remapping

Stock MiniMax H3 compacts only active references before assigning prompt ordinals.
This breaks fixed-role workflows when optional references are skipped.

New node:
**H3 Semantic Reference Director — Auto Remap**

It accepts fixed logical roles, compacts the active media, and rewrites logical
Picture/Video/Audio tags to the actual active H3 ordinals immediately before
tokenization.

Additional fixes:
- wardrobe top-crop control for references containing another person
- strict missing-tag detection
- H3-safe 32px resolution presets
- short delivery requests generate at least 124 frames before exact trim


## v0.4.1 — semantic prompt preflight fix

Fixes a Workflow 01 failure where the workflow's own role boilerplate could
mention inactive optional references and trigger strict_missing_tags.

Changes:
- missing-reference validation/remapping now happens BEFORE VAE encoding
- genuine bad tags fail immediately instead of after expensive reference encoding
- Workflow 01 v1.4.1 no longer puts the full optional-role dictionary inside
  the actual H3 creative prompt
- motion/video encoding path itself is unchanged from v0.4.0


## v0.5.0 - Workflow 01 v2 (additive)

- No hard `torchaudio` dependency at import time (newer ComfyUI builds do not ship it).
- `H3SemanticReferenceToVideo`: optional `prompt_prefix` input and a 5th `hq_ref_pack`
  output. Outputs 1-4 are unchanged.
- New nodes: **H3 LoRA Stack**, **H3 Speed Mode**, **H3 Cache Route**, **H3 HQ Gate**,
  **H3 HQ Pass-2 Latent**, **H3 HQ Pass-2 Conditioning**.
- The HQ pass needs the LBH latent upscaler node pack and a ComfyUI build with MiniMax H3
  audio-video noise-mask support. The companion installer checks both.


## v0.6.0 - ControlNet on Video 1 + face refine (additive)

- **H3 Video 1 Route**: Video 1 stays a normal reference until the workflow's ControlNet group is
  switched on; then it drives ComfyUI's core MiniMax H3 Fun ControlNet instead.
- **H3 Face Gate** and **H3 Face Audio Lock**: the switch and the audio lock for the optional
  face-refine branch (tracking and stitching come from ComfyUI-H3-FaceRefine).

## v0.6.1 - LoRA file names

H3 LoRA Stack now finds Cinematic Realism, Motion Continuity Repair and Better Human Motion by
their real file names. Cinematic Realism adds its trigger word "DY". Default strengths follow
the LoRA authors (Cinematic 0.7, Spatial Physics 0.4).

## v0.11.0 - Reference Director 01 v2.1

- **H3 LoRA Stack** has a new last field, `lora_apply_mode`:
  - `runtime` (default): each LoRA is applied while the model runs, with the calls of ComfyUI's core
    "Load LoRA (Bypass)" node.
  - `merged`: the LoRA is folded into the weights (the v0.10.1 behaviour).
  - All runtime LoRAs of one stack share one injection, undone in reverse order, so several LoRAs
    work together. The stack also works next to one core "Load LoRA (Bypass)" node or a second
    runtime stack, in any order. Two core Bypass nodes in one chain overwrite each other (ComfyUI
    core behaviour), so load several LoRAs in this stack instead.
- **H3 Speed Sigmas** (new): Speed Mode OFF samples with the full-quality schedule; the TURBO modes
  keep the plain schedule their LoRAs were made for.
- **H3 Veda Settings** (new): settings for the third-party "Veda Sparse Attention (MiniMax H3)"
  node. It keeps the late steps and every reference at full attention, so only the early steps are
  sparse.
- Workflows saved with v0.10.1 or older open with `lora_apply_mode` = `runtime`.

## v0.11.1 - Reference Director 01 v2.1

- **Upscale, motion and face switches** take an optional `run_after` video. Linked to the normal
  video, the optional groups start only after the normal video is saved, so it shows up first.

## v0.11.2 - Reference Director 01 v2.1.1

- **H3 LoRA Stack** has a new last field, `look_loras_below_noise` (default 1.0 = v2.1 behaviour).
  Below 1.0, the look LoRAs (Realism People, LMS Sharpness, custom 1 / 2) run only on the steps
  that produce noise below that level, where shapes and skin detail form; the early steps that
  decide framing, body structure and motion run the plain model. Motion Continuity Repair and
  Spatial Physics are never windowed. Runtime mode only. Start at 0.75 for medium and full-body
  shots (on the 24-step default render: off on steps 1-14, fading in on 15, full on 16-24; in the
  TURBO modes only the last step or two). The console prints the step count when a render starts.

## v0.11.3 - Reference Director 01 v2.1.3

- **H3 Keep Source Colors** (new): the upscaled video takes its colours from the normal video and
  keeps only the upscale's detail, so the HQ refine (which starts at 75 % noise) can no longer
  recolour thin bright details such as wet hair strands. Optional coarse-brightness lock.
