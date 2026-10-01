#!/usr/bin/env python3
"""
train_atlas.py — atlas-forge v0. Differentiable glyph atlas optimization.

Pre-registered in receipts/pre-registration.md (2026-10-01, before any run).
Pinned: seed=20261001, N=70, 4x6, Adam lr=0.01, epochs=3000.

Exports:
  exports/atlas_optimized.u32.json   — {"glyphs": [u32...], "ink": [...]}
  exports/atlas_optimized.js         — Uint32Array literal (engine drop-in)
  exports/training_trace.json        — loss components per 100 epochs
"""
import torch
import torch.nn as nn
import json, math, sys, time

SEED = 20261001
N, H, W = 70, 6, 4
EPOCHS = 3000
LR = 0.01
ALPHA, BETA, GAMMA, DELTA = 1.0, 0.5, 0.5, 0.05
NBINS = 16
MAG_FLOOR = 0.05
EPS = 1e-8

torch.manual_seed(SEED)

# Sobel kernels (1,1,H,W) for a 6x4 field, same weights as the engine ramp.
KX = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
KY = KX.t().contiguous()
kx = KX.view(1, 1, 3, 3)
ky = KY.view(1, 1, 3, 3)

# Assigned orientation bins: glyph i -> bin i mod 16 (pre-registered).
assigned_bin = torch.arange(N) % NBINS
bin_centers = (assigned_bin.float() + 0.5) * (2 * math.pi / NBINS)  # (N,)


def sobel_field(p):
    """p: (N,1,H,W) soft raster -> gx, gy (N,1,H,W), pad+shift (no conv dep)."""
    pad = torch.nn.functional.pad(p, (1, 1, 1, 1), mode="replicate")
    gx = (-pad[:, :, :-2, :-2] + pad[:, :, :-2, 2:]) \
         + 2.0 * (-pad[:, :, 1:-1, :-2] + pad[:, :, 1:-1, 2:]) \
         + (-pad[:, :, 2:, :-2] + pad[:, :, 2:, 2:])
    gy = (-pad[:, :, :-2, :-2] - 2.0 * pad[:, :, :-2, 1:-1] - pad[:, :, :-2, 2:]) \
         + (pad[:, :, 2:, :-2] + 2.0 * pad[:, :, 2:, 1:-1] + pad[:, :, 2:, 2:])
    return gx, gy


def main():
    t0 = time.time()
    profiles = nn.Parameter(torch.randn(N, 1, H, W))
    opt = torch.optim.Adam([profiles], lr=LR)
    target_ramps = torch.linspace(0.02, 0.98, N)
    eye_mask = 1.0 - torch.eye(N)
    trace = []

    lock_warned = False
    for epoch in range(EPOCHS):
        opt.zero_grad()
        p = torch.sigmoid(profiles)                      # (N,1,6,4) soft bits

        # --- L_density: sorted ink vs uniform ramp
        ink = p.mean(dim=(1, 2, 3))                      # (N,)
        l_density = torch.mean((torch.sort(ink)[0] - target_ramps) ** 2)

        # --- L_sobel: circular distance to assigned bin center
        gx, gy = sobel_field(p)                          # (N,1,H,W)
        mag = torch.sqrt(gx * gx + gy * gy + EPS)
        sin_w = (gy / mag).sum(dim=(1, 2, 3))            # (N,)
        cos_w = (gx / mag).sum(dim=(1, 2, 3))
        theta = torch.atan2(sin_w, cos_w) + math.pi      # (N,) in [0, 2π)
        d = theta - bin_centers
        circ = torch.atan2(torch.sin(d), torch.cos(d))   # wrapped distance
        mag_strength = torch.sqrt(sin_w ** 2 + cos_w ** 2)
        valid = (mag_strength > MAG_FLOOR).float()
        l_sobel = (circ.abs() * valid).sum() / valid.sum().clamp(min=1.0)

        # --- L_ortho: off-diagonal cosine similarity (normalized — the fix)
        flat = p.view(N, -1)                             # (N, 24)
        fn = torch.nn.functional.normalize(flat, dim=1, eps=EPS)
        cossim = fn @ fn.t()                             # (N,N)
        l_ortho = (cossim * eye_mask).pow(2).sum() / (N * (N - 1))

        # --- L_conf: bit confidence (away from 0.5)
        l_conf = (4.0 * p * (1.0 - p)).mean()

        loss = ALPHA * l_density + BETA * l_sobel + GAMMA * l_ortho + DELTA * l_conf
        loss.backward()
        opt.step()

        # RISK-3 check: premature bit-lock
        if epoch == 100 and not lock_warned:
            locked = ((p.detach() - 0.5).abs() > 0.45).float().mean().item()
            if locked > 0.95:
                lock_warned = True
                print(f"RISK-3 TRIGGERED at epoch 100: {locked:.2%} bits locked", file=sys.stderr)

        if epoch % 100 == 0 or epoch == EPOCHS - 1:
            trace.append({
                "epoch": epoch,
                "loss": round(loss.item(), 6),
                "density": round(l_density.item(), 6),
                "sobel": round(l_sobel.item(), 6),
                "ortho": round(l_ortho.item(), 6),
                "conf": round(l_conf.item(), 6),
            })

    # --- serialize: threshold >0.5, bit = row*4+col
    p_final = torch.sigmoid(profiles.detach()).squeeze(1)    # (N,6,4)
    bits = (p_final > 0.5)
    glyphs = []
    for g in range(N):
        v = 0
        for r in range(H):
            for c in range(W):
                if bits[g, r, c]:
                    v |= 1 << (r * 4 + c)
        glyphs.append(v)
    ink_out = [round(float(p_final[g].mean()), 4) for g in range(N)]

    with open("exports/atlas_optimized.u32.json", "w") as f:
        json.dump({"seed": SEED, "glyphs": glyphs, "ink": ink_out,
                   "epochs": EPOCHS, "loss_final": trace[-1]["loss"]}, f, indent=1)
    with open("exports/atlas_optimized.js", "w") as f:
        f.write("// atlas-forge v0 optimized atlas — engine drop-in (70 glyphs, 4x6, bit=row*4+col)\n")
        f.write(f"// seed={SEED} epochs={EPOCHS} loss={trace[-1]['loss']}\n")
        f.write("const FONT_ATLAS_DATA = new Uint32Array([" + ", ".join(f"0x{v:06x}" for v in glyphs) + "]);\n")
    with open("exports/training_trace.json", "w") as f:
        json.dump({"seed": SEED, "trace": trace,
                   "seconds": round(time.time() - t0, 1)}, f, indent=1)

    uniq = len(set(glyphs))
    print(f"trained {EPOCHS} epochs in {time.time()-t0:.1f}s  final loss {trace[-1]['loss']}")
    print(f"unique signatures: {uniq}/{N}")
    print("exports: atlas_optimized.u32.json, atlas_optimized.js, training_trace.json")


if __name__ == "__main__":
    main()
