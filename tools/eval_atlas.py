#!/usr/bin/env python3
"""
eval_atlas.py — A/B scene quantization benchmark, pre-registered metrics M1–M5.

Benchmark protocol is INDEPENDENT of the training code (RISK-2 mitigation):
same synthetic scene recipe as chiaroscuro tools/sobel_agree.py
(seed=20261001, 30 frames: diagonal/disk/noise x10, 480x360, 120x60 cells,
4x6 subpixels, pattern bit = subpixel luma > 0.5).

M5 per atlas = mean over cells of min-Hamming(true_pattern, elected_glyph)
where election = argmin Hamming over the atlas (the engine's XOR+POPCNT loop).

Atlases compared:
  A) LiberationMono-derived real atlas (font_atlas_packager 73045ae), first 70
     glyphs, re-serialized from the JS atlas order.
  B) atlas-forge optimized atlas (exports/atlas_optimized.u32.json).
  C) null control: seed=1234 random 70x24-bit codes (R4).

Verdict rule (receipts/pre-registration.md):
  SHIP iff M5(B) < M5(A) on >=2 of 3 scene classes AND unique(B) >= unique(A).
"""
import numpy as np
import json, math, re, sys

SEED = 20261001
W, H = 480, 360
COLS, ROWS = 120, 60
SUBW, SUBH = 4, 6

POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def load_chiaroscuro_atlas():
    src = open("/root/.openclaw/workspace/repos/chiaroscuro/js/font_atlas.js").read()
    i = src.find("FONT_ATLAS_DATA")
    m = re.match(r"\s*FONT_ATLAS_DATA\s*=\s*new Uint32Array\(\[([^\]]+)\]\)", src[i:])
    arr = [int(x.strip(), 0) for x in m.group(1).split(",") if x.strip()]
    return np.array(arr[:70], dtype=np.uint32)          # first 70, engine order


def load_optimized_atlas():
    d = json.load(open("/root/.openclaw/workspace/repos/atlas-forge/exports/atlas_optimized.u32.json"))
    return np.array(d["glyphs"], dtype=np.uint32)


def load_null_atlas():
    rng = np.random.default_rng(1234)
    return rng.integers(0, 1 << 24, size=70, dtype=np.uint32)


def popcount_u32(arr):
    b = arr.view(np.uint8).reshape(arr.shape + (4,))
    return POP[b].sum(axis=-1)


def mean_min_hamming(sigs, atlas):
    """sigs: (N,) uint32 cell patterns -> mean min-Hamming to atlas."""
    x = sigs[:, None] ^ atlas[None, :]
    pc = popcount_u32(x.reshape(-1)).reshape(x.shape)
    return float(pc.min(axis=1).mean())


def luma_field(kind, rng, angle=None):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    if kind == "diagonal":
        a = angle if angle is not None else rng.uniform(0, math.pi)
        f = np.sin((xx * math.cos(a) + yy * math.sin(a)) / 12.0)
        return (f > 0).astype(np.float64) * 0.8 + 0.1
    if kind == "disk":
        cx, cy = rng.uniform(100, 380), rng.uniform(80, 280)
        r = rng.uniform(30, 120)
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        return np.where(d < r, 0.9, 0.1).astype(np.float64)
    return rng.random((H, W))


def cell_patterns(gray):
    bits = (gray > 0.5).astype(np.uint32)
    packed = bits.reshape(ROWS, SUBH, COLS, SUBW)
    sig = np.zeros((ROWS, COLS), dtype=np.uint32)
    for y in range(SUBH):
        for x in range(SUBW):
            sig |= (packed[:, y, :, x] << np.uint32(y * 4 + x))
    return sig.reshape(-1)


def metrics(atlas):
    uniq = len(set(atlas.tolist()))
    # M1 min pairwise Hamming
    x = atlas[:, None] ^ atlas[None, :]
    pc = popcount_u32(x.reshape(-1)).reshape(x.shape).astype(np.int32)
    np.fill_diagonal(pc, 99)
    m1 = int(pc.min())
    # M3 ink ramp R^2 (thresholded bits)
    ink = np.array([bin(v).count("1") for v in atlas], dtype=np.float64) / 24.0
    s_ink = np.sort(ink)
    ramp = np.linspace(0.02, 0.98, len(atlas))
    ss_res = float(((s_ink - ramp) ** 2).sum())
    ss_tot = float(((ramp - ramp.mean()) ** 2).sum())
    m3 = 1.0 - ss_res / ss_tot
    return {"unique": uniq, "min_hamming": m1, "ink_ramp_r2": round(m3, 4)}


def main():
    atlases = {
        "liberation_mono": load_chiaroscuro_atlas(),
        "optimized": load_optimized_atlas(),
        "null_random": load_null_atlas(),
    }
    rng = np.random.default_rng(SEED)
    scenes = [("diagonal", 10), ("disk", 10), ("noise", 10)]
    frames = {k: [] for k, _ in scenes}
    for kind, n in scenes:
        for _ in range(n):
            frames[kind].append(cell_patterns(luma_field(kind, rng)))

    out = {"seed": SEED, "atlases": {}, "m5_scene_quantization": {}}
    for name, atlas in atlases.items():
        out["atlases"][name] = metrics(atlas)
        out["m5_scene_quantization"][name] = {}
        for kind, _ in scenes:
            m = np.mean([mean_min_hamming(f, atlas) for f in frames[kind]])
            out["m5_scene_quantization"][name][kind] = round(float(m), 4)

    # verdict per pre-registered rule
    A = out["m5_scene_quantization"]["liberation_mono"]
    B = out["m5_scene_quantization"]["optimized"]
    wins = sum(1 for k in ("diagonal", "disk", "noise") if B[k] < A[k])
    uniq_ok = out["atlases"]["optimized"]["unique"] >= out["atlases"]["liberation_mono"]["unique"]
    out["verdict"] = {
        "m5_wins_vs_liberation": wins,
        "uniqueness_ok": uniq_ok,
        "rule": "SHIP iff wins>=2 AND uniqueness_ok",
        "decision": "SHIP" if (wins >= 2 and uniq_ok) else "DO_NOT_SHIP",
    }
    print(json.dumps(out, indent=2))
    with open("/root/.openclaw/workspace/repos/atlas-forge/receipts/eval_receipt.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nreceipt -> receipts/eval_receipt.json")


if __name__ == "__main__":
    main()
