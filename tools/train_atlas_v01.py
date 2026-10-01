#!/usr/bin/env python3
"""
train_atlas_v01.py — atlas-forge v0.1: distance-aware training.

Delta vs v0 (receipts/pre-registration-v01.md, sealed before this run):
  gamma 0.5->1.5, delta annealed 0.05->0.0, new L_dist hinge toward
  Hamming >= 8 (lambda=0.3). Everything else pinned identical to v0.
"""
import torch
import torch.nn as nn
import json, math, sys, time

SEED = 20261001
N, H, W = 70, 6, 4
EPOCHS = 3000
LR = 0.01
ALPHA, BETA, GAMMA = 1.0, 0.5, 1.5
LAMBDA_DIST, D_TARGET = 0.3, 8
NBINS = 16
MAG_FLOOR = 0.05
EPS = 1e-8

torch.manual_seed(SEED)
assigned_bin = torch.arange(N) % NBINS
bin_centers = (assigned_bin.float() + 0.5) * (2 * math.pi / NBINS)

II, JJ = torch.triu_indices(N, N, offset=1)   # 2415 pairs


def sobel_field(p):
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

    for epoch in range(EPOCHS):
        opt.zero_grad()
        p = torch.sigmoid(profiles)

        ink = p.mean(dim=(1, 2, 3))
        l_density = torch.mean((torch.sort(ink)[0] - target_ramps) ** 2)

        gx, gy = sobel_field(p)
        mag = torch.sqrt(gx * gx + gy * gy + EPS)
        sin_w = (gy / mag).sum(dim=(1, 2, 3))
        cos_w = (gx / mag).sum(dim=(1, 2, 3))
        theta = torch.atan2(sin_w, cos_w) + math.pi
        d = theta - bin_centers
        circ = torch.atan2(torch.sin(d), torch.cos(d))
        mag_strength = torch.sqrt(sin_w ** 2 + cos_w ** 2)
        valid = (mag_strength > MAG_FLOOR).float()
        l_sobel = (circ.abs() * valid).sum() / valid.sum().clamp(min=1.0)

        flat = p.view(N, -1)
        fn = torch.nn.functional.normalize(flat, dim=1, eps=EPS)
        cossim = fn @ fn.t()
        l_ortho = (cossim * eye_mask).pow(2).sum() / (N * (N - 1))

        # L_dist: soft-agreement hinge toward Hamming >= D_TARGET
        pi = flat[II]                                  # (P, 24)
        pj = flat[JJ]
        agree = (pi * pj + (1 - pi) * (1 - pj)).sum(dim=1)   # (P,)
        l_dist = torch.relu(agree - (24 - D_TARGET)).mean()

        # annealed confidence
        delta = 0.05 * (1.0 - epoch / max(EPOCHS - 1, 1))
        l_conf = (4.0 * p * (1.0 - p)).mean()

        loss = (ALPHA * l_density + BETA * l_sobel + GAMMA * l_ortho
                + delta * l_conf + LAMBDA_DIST * l_dist)
        loss.backward()
        opt.step()

        if epoch % 100 == 0 or epoch == EPOCHS - 1:
            trace.append({"epoch": epoch, "loss": round(loss.item(), 6),
                          "density": round(l_density.item(), 6),
                          "sobel": round(l_sobel.item(), 6),
                          "ortho": round(l_ortho.item(), 6),
                          "dist": round(l_dist.item(), 6),
                          "delta": round(delta, 5)})

    p_final = torch.sigmoid(profiles.detach()).squeeze(1)
    bits = (p_final > 0.5)
    glyphs = []
    for g in range(N):
        v = 0
        for r in range(H):
            for c in range(W):
                if bits[g, r, c]:
                    v |= 1 << (r * 4 + c)
        glyphs.append(v)

    base = "/root/.openclaw/workspace/repos/atlas-forge/exports/"
    with open(base + "atlas_v01.u32.json", "w") as f:
        json.dump({"seed": SEED, "glyphs": glyphs, "epochs": EPOCHS,
                   "loss_final": trace[-1]["loss"], "version": "v0.1"}, f, indent=1)
    with open(base + "atlas_v01.js", "w") as f:
        f.write("// atlas-forge v0.1 distance-aware atlas — engine drop-in\n")
        f.write(f"// seed={SEED} epochs={EPOCHS} loss={trace[-1]['loss']} d_target={D_TARGET}\n")
        f.write("const FONT_ATLAS_DATA = new Uint32Array([" + ", ".join(f"0x{v:06x}" for v in glyphs) + "]);\n")
    with open(base + "training_trace_v01.json", "w") as f:
        json.dump({"seed": SEED, "trace": trace,
                   "seconds": round(time.time() - t0, 1)}, f, indent=1)

    uniq = len(set(glyphs))
    print(f"v0.1 trained {EPOCHS} epochs in {time.time()-t0:.1f}s  final loss {trace[-1]['loss']}")
    print(f"unique: {uniq}/{N}")


if __name__ == "__main__":
    main()
