"""
Maxwell 3D BI-CIS pixel — IPA light trapping + DTI crosstalk suppression
========================================================================
The full back-illuminated pixel of Yokogawa et al. (Sci. Rep. 7:3832, 2017 —
an IR-enhanced CMOS image sensor): the complete stack of the paper's
Figs. 3/4/7. Companion to ``maxwell_ipa_light_trapping.py``, which solves the
bare surface unit cell (paper Fig. 2).

Per 1.2 μm pixel (paper Fig. 3, top to bottom):
  * on-chip microlens (OCL): spherical cap, curvature radius R_c = 0.85 μm,
    lens stack height 1.7 μm from the Si surface to the lens top;
  * planarization / filter spacer (polymer, n ≈ 1.6). NO color filters — this is
    an NIR (850 nm) study and the paper's own NIR data is effectively filter-free;
  * 3 μm crystalline-Si photodiode. Its top surface is the variable:
      Sample A — flat            (reference)
      Sample B — 3x3 inverted-pyramid array (IPA), 400 nm pitch
      Sample C — IPA + deep trench isolation (DTI): SiO2-filled trenches,
                 200 nm wide x 2 μm deep, along every pixel border.
  * below the photodiode: absorbing boundary (transmitted light never returns —
    conservative; the real wiring stack can diffract some of it back).

Two experiments, matching the paper's two claims:

PART 1 — sensitivity (paper Fig. 4): uniform illumination on a periodic pixel
array (one-pixel unit cell). Reports the absorbed fraction in the 3 μm
photodiode per sample and the enhancement FOM = A_sample / A_flat.
EXPECTED OUTPUT:  A = 11.7%,  B = 14.4% (FOM 1.24),  C = 15.0% (FOM 1.29).
These numbers are properties of THIS simplified model, not device predictions:
the model ends in a perfect absorber below the photodiode (no wiring-stack
reflections back into the Si), Sample A carries no AR passivation, and there
are no color filters. The paper's device-level figures include exactly those
omitted effects, so they are not directly comparable — the point of this
example is the workflow (build the stack, run, measure per-photodiode
absorption), and every omission above is a straightforward extension of the
same medium-building code.

PART 2 — optical crosstalk (paper Figs. 6/7): an N_PIX_XTALK x N_PIX_XTALK pixel
array (default 5x5 — two full rings of neighbours) with absorbing side
boundaries, ONLY the centre pixel illuminated (apodized Gaussian, FWHM =
one pixel — same methodology as maxwell_microlens.py). The absorbed power per
photodiode gives the per-pixel signal matrix:
  * Sample B: the IPA diffracts light sideways -> neighbours light up (the
    paper's blurred NIR images and washed-out colors);
  * Sample C: the DTI totally-internally-reflects those rays back into the
    pixel -> same central signal, neighbours dark (the paper's sharp images).
Ray tracing cannot produce ANY of this — the diffraction is the mechanism.
EXPECTED OUTPUT (5x5 default; ~380 s/solve, ~30 GB of GPU memory):
neighbour-total / centre = 45.0% for A (mostly the Gaussian illumination
tails — the unavoidable baseline), 130.3% for B, 53.7% for C. The central 3x3
block is independent of the array size (a 3x3 run gives 41.3 / 100.1 / 46.2 —
a built-in consistency check). B leaks well into the SECOND ring — up to 5.8%
of the centre signal two pixels away — while C's second ring stays near the
flat baseline. Two details worth noticing in the output: (1) the source is
x-polarized, so B leaks hardest into ±x (27.5% each vs 11.9% in ±y);
unpolarized light would symmetrize the split, not the totals. (2) C's first
ring sits slightly above flat (10.3% vs 9.1%) because the DTI stops 2 μm deep
and some light crosses beneath the trench — visible as the deep lobe in the C
field slice.

Cost note: Part 1 cells are tiny (60x60x300 voxels). Part 2 is the heavyweight
showcase: at the 5x5 default, a 300x300x300 stated domain and ~380 Mvoxel
including the absorbing boundaries the solver adds outside it. Typical cost:
~380 s per solve and ~30 GB of GPU memory. For scale: the 6 μm pixel-array
domain alone at the 4 nm mesh the reference paper used for its pixel FDTD is
~3.4 GVOXEL (125x), and ~27 GVOXEL at the 2 nm of its unit-cell study — here
it runs on one GPU in six minutes. Monitors return only the photodiode Si block
(~330 MB at 5x5) and a thin viz slice, not the full volume. Set N_PIX_XTALK = 3
for the lighter variant or RUN_CROSSTALK = False to run Part 1 alone.

Run from the repo root (see CLAUDE.md):
    python examples/maxwell_ipa_pixel.py
Outputs land in the git-ignored output/ folder.
"""

import os
import signal
from time import time

import numpy as np
import matplotlib.pyplot as plt

from wavesimpro import configure, create_monitor, simulate
from wsutils import OUTPUT_DIR, plane_wave
from wsutils.create_source import gaussian_beam

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Parameters
# =============================================================================
wavelength = 0.850    # μm — the paper's headline NIR wavelength
pixel_size = 0.020    # μm — λ/12 in Si at 850 nm (MBS needs only λ/4–λ/8);
                      # a 10 nm convergence check is identical
boundary_um = 4.25    # absorbing boundary = 5λ (MBS sizing rule) — sufficient at 850 nm
                      # because the Si itself damps the diffracted orders before they
                      # reach the bottom absorber

RUN_CROSSTALK = True  # Part 2 (3 heavyweight runs); Part 1 always runs first
N_PIX_XTALK = 5       # pixels per side in the Part 2 array (odd). 5 shows TWO rings of
                      # crosstalk decay (~380 Mvoxel with boundaries, Si monitor ~330 MB);
                      # 3 is the lighter variant.
SAMPLES = ["A", "B", "C"]

# --- Materials ----------------------------------------------------------------
n_air = 1.0
n_lens = 1.60         # OCL + planarization/filter-spacer polymer
n_sio2 = 1.45         # pyramid backfill + DTI fill
n_si = 3.653 + 0.00362j  # c-Si at 850 nm (Green 2008; k = alpha*lambda/4pi, alpha=535/cm)

# --- Pixel geometry (paper Fig. 3) ---------------------------------------------
pix = 1.20            # pixel pitch (μm)
ipa_pitch = 0.400     # 3x3 pyramids per pixel
facet_deg = 54.7      # Si(111) facet angle -> pyramid depth = (pitch/2)*tan(54.7°)
dti_width = 0.200     # DTI width (μm), Sample C
dti_depth = 2.000     # DTI depth from the Si surface (μm), Sample C
R_c = 0.85            # OCL curvature radius (μm)
lens_stack = 1.70     # Si surface -> lens top (μm)

# --- z-stack (z = propagation/depth; x, y in-plane) ----------------------------
z_src = 0.25          # source plane in the air above the lenses (μm)
z_vertex = 0.60       # OCL cap vertex (top of the lens)
z_si0 = z_vertex + lens_stack       # Si surface = 2.30 μm
si_thick = 3.0        # photodiode thickness = absorption metric depth (μm)
z_end = z_si0 + si_thick + 0.7      # small Si margin, then the bottom absorber
nz = int(round(z_end / pixel_size))
iz_si0 = int(round(z_si0 / pixel_size))
n_metric = int(round(si_thick / pixel_size))          # 150 z-voxels of photodiode
iz_dti1 = int(round((z_si0 + dti_depth) / pixel_size))

sag = R_c - np.sqrt(R_c ** 2 - (pix / 2) ** 2)        # OCL cap height (≈0.25 μm)
z_sphere = z_vertex + R_c                             # OCL defining-sphere centre
iz_vertex = int(round(z_vertex / pixel_size))
iz_base = int(round((z_vertex + sag) / pixel_size))   # cap base -> slab below

npp = int(round(pix / pixel_size))                    # voxels per pixel (60, even)
pyr_depth = ipa_pitch / 2 * np.tan(np.deg2rad(facet_deg))

# Monitor z-extents (odd voxel counts -> exact grid alignment)
M_SI = 2 * ((n_metric + 1) // 2) + 1                  # photodiode block + margin
IZ_SLICE0 = int(round(0.4 / pixel_size))              # viz slice from 0.4 μm down
_m = nz - 1 - IZ_SLICE0
M_SLICE = _m if _m % 2 else _m - 1
assert iz_si0 + M_SI <= nz


# =============================================================================
# Medium builder: n_pix x n_pix pixels of Sample A / B / C
# =============================================================================
def build(sample, n_pix):
    """Permittivity (n²), shape (nxy, nxy, nz) complex64, and the apex/centre
    viz row. All features fold to nearest-centre distances, so any n_pix works."""
    nxy = n_pix * npp
    xy = (np.arange(nxy) + 0.5) * pixel_size
    d_pix = np.abs(xy - (np.floor(xy / pix) + 0.5) * pix)       # to pixel centre
    d_pyr = np.abs(xy - (np.floor(xy / ipa_pitch) + 0.5) * ipa_pitch)  # to pyramid centre
    d_border = np.abs(xy - np.round(xy / pix) * pix)            # to pixel border

    n = np.full((nxy, nxy, nz), n_air, dtype=np.complex64)

    # OCL: spherical cap per pixel (vertex up), then the planar polymer slab.
    r2_pix = d_pix[:, None] ** 2 + d_pix[None, :] ** 2
    for iz in range(iz_vertex, iz_base + 1):
        dz = z_sphere - (iz + 0.5) * pixel_size
        if dz >= R_c:
            continue
        n[r2_pix < R_c ** 2 - dz ** 2, iz] = n_lens
    n[:, :, iz_base:iz_si0] = n_lens

    # Photodiode.
    n[:, :, iz_si0:] = n_si

    if sample in ("B", "C"):   # IPA: SiO2-refilled inverted pyramids
        cheb = np.maximum(d_pyr[:, None], d_pyr[None, :])
        for iz in range(iz_si0, iz_si0 + int(np.ceil(pyr_depth / pixel_size))):
            w = ipa_pitch / 2 * (1 - ((iz - iz_si0 + 0.5) * pixel_size) / pyr_depth)
            if w <= 0:
                break
            n[cheb < w, iz] = n_sio2

    if sample == "C":          # DTI: SiO2 trenches along every pixel border
        dti = (d_border < dti_width / 2)[:, None] | (d_border < dti_width / 2)[None, :]
        n[dti, iz_si0:iz_dti1] = n_sio2

    return (n ** 2).astype(np.complex64), nxy, npp // 2


# =============================================================================
# Solve + measure helpers (same monitor pattern as maxwell_ipa_light_trapping.py)
# =============================================================================
def zcenter(iz0, m):
    return (iz0 + (m - 1) / 2 + 0.5 - nz / 2) * pixel_size


def solve(permittivity, nxy, source, periodic):
    si_block = create_monitor(
        name="si_block", center_x=0.0, center_y=0.0, center_z=zcenter(iz_si0, M_SI),
        size_x=nxy * pixel_size, size_y=nxy * pixel_size, size_z=M_SI * pixel_size)
    iy = nxy // 2  # a row through the CENTRE pixel's midline (also an apex row)
    xz_slice = create_monitor(
        name="xz_slice", center_x=0.0, center_y=(iy + 0.5 - nxy / 2) * pixel_size,
        center_z=zcenter(IZ_SLICE0, M_SLICE),
        size_x=nxy * pixel_size, size_y=pixel_size, size_z=M_SLICE * pixel_size)
    u, iterations, residual = simulate(
        permittivity=permittivity,
        sources=[source],
        wavelength=wavelength,
        pixel_size=pixel_size,
        boundary_width=boundary_um,
        periodic=periodic,
        use_gpu=True,
        monitors=[si_block, xz_slice],
    )
    esq_si = (np.abs(u[0]) ** 2).sum(axis=0)
    assert esq_si.shape == (nxy, nxy, M_SI), \
        f"Si monitor returned {esq_si.shape}, expected {(nxy, nxy, M_SI)}"
    mag = np.sqrt((np.abs(u[1]) ** 2).sum(axis=0))[:, 0, :].astype(np.float32)
    return esq_si, mag, iterations, residual


def absorbed_density(esq_si, permittivity):
    """Im(ε)|E|² over the photodiode metric depth, shape (nxy, nxy, n_metric)."""
    return permittivity.imag[:, :, iz_si0:iz_si0 + n_metric] * esq_si[:, :, :n_metric]


configure(description="maxwell_ipa_pixel")
k0 = 2 * np.pi / wavelength


# =============================================================================
# PART 1 — sensitivity: periodic pixel array, uniform illumination
# =============================================================================
print(f"PART 1 — periodic pixel, uniform illumination, lambda = {wavelength*1e3:.0f} nm")
nxy1 = npp
src_shape = [(nxy1 + 0.5) * pixel_size] * 2
pw_values, pw_position = plane_wave(
    shape=src_shape, origin="topleft", position=[0, 0.0, 0.0, z_src],
    source_plane="xy", pixel_size=pixel_size)

# |E0|² calibration: identical source in a uniform-air domain (incidence medium).
eps_air = np.full((nxy1, nxy1, nz), n_air ** 2, dtype=np.complex64)
esq_ref, _, _, _ = solve(eps_air, nxy1, (pw_values, pw_position), (True, True, False))
e0sq = float(esq_ref.mean())

part1 = {}      # sample -> absorbed fraction
slices1 = {}    # sample -> |E| viz slice
for sample in SAMPLES:
    permittivity, nxy, _ = build(sample, 1)
    start = time()
    esq_si, mag, iters, res = solve(permittivity, nxy, (pw_values, pw_position),
                                    (True, True, False))
    A = k0 * pixel_size * absorbed_density(esq_si, permittivity).sum() \
        / (n_air * e0sq * nxy * nxy)
    part1[sample] = A
    slices1[sample] = mag
    fom = A / part1["A"]
    print(f"  Sample {sample}:  A = {A*100:5.2f}%   FOM = {fom:4.2f}   "
          f"({iters} iters, res {res:.1e}, {time()-start:.1f} s)")
print("  (Simplified model — perfect absorber below the photodiode, no AR layer,"
      " no color filters: figures describe this model, not the physical device.)")


# =============================================================================
# PART 2 — crosstalk: 3x3 pixels, ONLY the centre pixel illuminated
# =============================================================================
signals = {}    # sample -> n_pix x n_pix absorbed-power matrix
slices2 = {}    # sample -> |E| viz slice through the centre-pixel midline
if RUN_CROSSTALK:
    n_pix = N_PIX_XTALK
    cpx = n_pix // 2                # centre-pixel index
    nxy2 = n_pix * npp
    # Apodized Gaussian on the centre lens, FWHM = one pixel: neighbours receive
    # essentially no direct light, so any neighbour signal IS optical crosstalk.
    m_src = nxy2
    sigma_px = (pix / pixel_size) / 2.3548          # FWHM = one pixel pitch
    gb_values, gb_position = gaussian_beam(
        shape=[(nxy2 + 0.5) * pixel_size] * 2, origin="topleft",
        position=[0, 0.0, 0.0, z_src], source_plane="xy",
        pixel_size=pixel_size, alpha=(m_src - 1) / (2 * sigma_px))

    print(f"\nPART 2 — {n_pix}x{n_pix} pixels, centre-only Gaussian (FWHM = {pix} um), "
          f"absorbing boundaries")
    print(f"  stated domain {nxy2}x{nxy2}x{nz} voxels; solver adds ~{boundary_um} um "
          f"absorbers on all six faces")
    for sample in SAMPLES:
        permittivity, nxy, _ = build(sample, n_pix)
        start = time()
        esq_si, mag, iters, res = solve(permittivity, nxy, (gb_values, gb_position),
                                        (False, False, False))
        dens = absorbed_density(esq_si, permittivity)
        # per-pixel absorbed power: fold the (nxy, nxy) plane into 3x3 pixels
        s = dens.reshape(n_pix, npp, n_pix, npp, n_metric).sum(axis=(1, 3, 4))
        signals[sample] = s
        slices2[sample] = mag
        xtalk = (s.sum() - s[cpx, cpx]) / s[cpx, cpx] * 100
        print(f"  Sample {sample}:  centre signal {s[cpx, cpx]:.3e}   "
              f"neighbour total = {xtalk:5.1f}% of centre   "
              f"({iters} iters, res {res:.1e}, {time()-start:.1f} s)")
    b, c = signals.get("B"), signals.get("C")
    if b is not None and c is not None:
        print(f"  DTI effect: neighbour/centre  B = "
              f"{(b.sum()-b[cpx,cpx])/b[cpx,cpx]*100:.1f}%  ->  C = "
              f"{(c.sum()-c[cpx,cpx])/c[cpx,cpx]*100:.1f}%   (same centre-pixel gain kept)")


# =============================================================================
# Restore SIGINT before plotting (cloud client can leave it wedged)
# =============================================================================
try:
    signal.signal(signal.SIGINT, signal.default_int_handler)
except (ValueError, TypeError):
    pass


# =============================================================================
# Figure 1 — Part 1: field slices + absorption/FOM bars
# =============================================================================
z0, z1 = IZ_SLICE0 * pixel_size, (IZ_SLICE0 + M_SLICE) * pixel_size
vmax1 = max(m.max() for m in slices1.values())
fig, axes = plt.subplots(1, len(SAMPLES) + 1, figsize=(4 * len(SAMPLES) + 4.5, 5.5),
                         constrained_layout=True)
titles = {"A": "A — flat", "B": "B — IPA", "C": "C — IPA + DTI"}
for ax, sample in zip(axes[:-1], SAMPLES):
    mag = np.tile(slices1[sample], (3, 1))          # show 3 periods
    extent = [0, 3 * pix, z1, z0]
    ax.imshow(mag.T, cmap="inferno", extent=extent, vmin=0, vmax=vmax1, aspect="auto")
    ax.axhline(z_si0, color="cyan", lw=0.8, ls="--", alpha=0.8)
    ax.set_title(titles[sample])
    ax.set_xlabel(r"$x~(\mu m)$")
    ax.set_ylabel(r"$z~(\mu m)$ — depth")
ax = axes[-1]
vals = [part1[s] * 100 for s in SAMPLES]
bars = ax.bar([titles[s] for s in SAMPLES], vals,
              color=["tab:gray", "tab:orange", "tab:green"])
for bar, s in zip(bars, SAMPLES):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            f"{part1[s]*100:.1f}%\nFOM {part1[s]/part1['A']:.2f}",
            ha="center", fontsize=9)
ax.set_ylabel("absorbed in 3 μm photodiode (%)")
ax.set_title("sensitivity @ 850 nm\n(uniform illumination, periodic array)")
ax.grid(True, axis="y", alpha=0.3)
fig.suptitle("BI-CIS pixel, λ = 850 nm — microlens + IPA light trapping "
             "(cyan = Si surface)", fontsize=11)
fig.savefig(f"{OUTPUT_DIR}/maxwell_ipa_pixel_sensitivity.png", dpi=300,
            bbox_inches="tight")

# =============================================================================
# Figure 2 — Part 2: crosstalk field slices + per-pixel signal matrices
# =============================================================================
if RUN_CROSSTALK and signals:
    fig2, axes2 = plt.subplots(2, len(SAMPLES), figsize=(4.6 * len(SAMPLES), 9.5),
                               constrained_layout=True)
    vmax2 = max(m.max() for m in slices2.values())
    for j, sample in enumerate(SAMPLES):
        ax = axes2[0, j]
        extent = [0, n_pix * pix, z1, z0]
        ax.imshow(slices2[sample].T, cmap="inferno", extent=extent,
                  vmin=0, vmax=vmax2, aspect="auto")
        ax.axhline(z_si0, color="cyan", lw=0.8, ls="--", alpha=0.8)
        for k in range(1, n_pix):
            ax.axvline(k * pix, color="w", lw=0.6, ls=":", alpha=0.6)
        ax.set_title(titles[sample])
        ax.set_xlabel(r"$x~(\mu m)$")
        ax.set_ylabel(r"$z~(\mu m)$ — depth")

        ax = axes2[1, j]
        centre = signals[sample][cpx, cpx]
        s = signals[sample] / centre * 100  # % of centre
        im = ax.imshow(s, cmap="viridis", vmin=0, vmax=100)
        for (r, cix), v in np.ndenumerate(s):
            ax.text(cix, r, f"{v:.1f}", ha="center", va="center",
                    color="w" if v < 60 else "k", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
        xt = (signals[sample].sum() - centre) / centre
        ax.set_title(f"per-pixel signal (% of centre)\nneighbour total = {xt*100:.1f}%")
    fig2.suptitle("Optical crosstalk, centre pixel lit only — IPA diffracts into "
                  "neighbours (B); DTI traps it (C)", fontsize=11)
    fig2.savefig(f"{OUTPUT_DIR}/maxwell_ipa_pixel_crosstalk.png", dpi=300,
                 bbox_inches="tight")

print(f"\nSaved {OUTPUT_DIR}/maxwell_ipa_pixel_sensitivity.png"
      + (" and maxwell_ipa_pixel_crosstalk.png" if RUN_CROSSTALK else ""))
plt.show()
