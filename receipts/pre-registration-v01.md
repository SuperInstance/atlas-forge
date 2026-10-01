# v0.1 Pre-Registration — distance-aware training

Date: 2026-10-01. Sealed BEFORE the v0.1 run (R2). v0 receipt:
receipts/eval_receipt.json (v0 SHIP, min Hamming 2 — RISK-1 surprise).

## Changes vs v0 (each tied to a v0 finding)

1. γ (ortho cosine): 0.5 → **1.5** — v0's soft cosine let near-duplicate
   hard codes through.
2. δ (bit confidence): 0.05 → **annealed 0.05→0.0 linear** over training —
   v0's fixed δ fought the distance gradient late in training (it pushes
   toward saturation regardless of code-space geometry).
3. NEW L_dist (λ=0.3): pairwise soft-agreement hinge toward Hamming ≥8.
   A_ij = Σ_k [p_i,k·p_j,k + (1−p_i,k)(1−p_j,k)]  (expected #equal bits)
   L_dist = mean_{i<j} relu(A_ij − (24 − 8))
   Straight-through-free: computed on soft bits; gradient flows through
   both glyphs of each pair. 2415 pairs → cheap.
   d_target=8 chosen from code-table feasibility (A(24,8) ≫ 70; RISK-1's
   6–10 band), NOT tuned post-hoc to results.

## Unchanged (pinned)

seed=20261001, N=70, 4×6, Adam lr=0.01, epochs=3000, α=1.0, β=0.5,
MAG_FLOOR=0.05, NBINS=16, bin assignment i mod 16.

## Verdict gates (unchanged from v0 — no moving goalposts)

- G1: M5(v0.1) < M5(LiberationMono) on ≥2/3 scene classes
- G2: unique(v0.1) ≥ unique(LiberationMono)   (70 ≥ 64 expected)
- REPORT-ONLY aspiration: M1 ≥ 6. Not a gate — if M1 lands <6 but G1+G2
  hold, v0.1 still ships and the distance gap is recorded honestly.

## Selection rule for engine integration

The atlas that goes into chiaroscuro must satisfy G1+G2 and have the best
M5 mean across classes among {v0, v0.1}. Tie → higher M1.
