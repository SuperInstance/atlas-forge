# atlas-forge — differentiable glyph atlas optimizer

Shape-first font generation as inverse template matching: evolve a 70-glyph
4x6 (24-bit) atlas by gradient descent so the glyph set maximizes coverage
ramp, orientation diversity, and pairwise distinctiveness — then serialize to
u32 signatures that drop into the chiaroscuro election engine contract
(73-glyph atlas slot, bit = row*4+col, XOR+POPCNT, WGSL no-float loop).

Lineage: font_atlas_packager (real ttf ray cast, 66/73 unique) → this repo
(evolved atlas) → A/B against both on the pinned synthetic scenes
(chiaroscuro tools/sobel_agree.py protocol, seed 20261001).

## Status

- v0 pre-registration: receipts/pre-registration.md (losses, metrics, verdict rule)
- v0 training: tools/train_atlas.py
- A/B evaluation: tools/eval_atlas.py (shared benchmark, both atlases)
- exports/: serialized u32 atlases + JS snippet

## Rules (carried from fleet R1–R8)

Receipts over claims. Seeds pinned before runs. Pre-register metrics and
verdict rules before training. Null control must be able to widen the gap.
No unmeasurable builds.

