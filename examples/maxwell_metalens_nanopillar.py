"""
Large-area nanopillar metalens — calibrated meta-atom library
=============================================================
A companion to maxwell_metalens.py that realizes the SAME hyperbolic focusing
phase with a genuine **sub-wavelength nanopillar metasurface** instead of a
surface relief. Each cell of a sub-wavelength square lattice holds one TiO₂
pillar; the pillar DIAMETER sets the local transmitted phase. This is the
canonical metasurface flat lens — distinct from the diffractive/relief optic in
maxwell_metalens.py.

The phase a pillar imprints is NOT modelled by effective medium here (that thin-
element approximation does not hold for these high-index meta-atoms). Instead we
**calibrate** it: solve a periodic unit cell (one pillar, periodic transverse
boundaries = the standard local-periodic approximation) for a sweep of diameters
and measure the actual transmitted phase. That measured diameter→phase map is then
inverted to place pillars realizing the lens phase. This is how real metasurfaces
are designed.

Conventions:
  * source axis order (polarization, x, y, z); a yz sheet is (1, 1, ny, nz);
  * y-polarization (transverse to +x propagation);
  * focusing-phase sign = −1 (φ(r) = −k₀(√(r²+f²) − f)).

Pipeline: (1) calibrate the library with small unit-cell solves; (2) guard — abort
before the expensive full solve if the library can't span 2π; (3) build the pillar
forest from the measured map; (4) solve the full 3-D lens and plot the focal cone
and Airy spot. Outputs mirror maxwell_metalens.py (metalens_nanopillar_*.png/.npy).
"""

import os
import signal
import sys
from time import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.signal.windows import tukey

from wsutils import OUTPUT_DIR
from wavesimpro import simulate

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Parameters ---------------------------------------------------------------
wavelength = 0.532
pixel_size = 0.03
n_pillar = 2.4 + 0.02j   # TiO₂-like; tiny absorption keeps the periodic unit cell
#                          (and the dense pillar array) well-conditioned for the MBS solver.
n_substrate = 1.5
n_bg = 1.0

aperture = 10.0          # circular aperture diameter, μm
focal_length = 12.0      # design focal length from the pillar exit plane, μm
lattice_pitch = 0.25     # square-lattice pitch, μm (sub-wavelength: < λ, no higher orders in air)
pillar_height = 0.7      # pillar height H, μm (tall enough for the library to span ≥ 2π)
focusing_sign = -1.0

k0 = 2 * np.pi / wavelength
NA = (aperture / 2) / np.sqrt((aperture / 2) ** 2 + focal_length ** 2)
dl_fwhm = 0.51 * wavelength / NA

# Domain layout along X: air gap → substrate → pillar layer → air focal region.
x_src = 0.6
x_sub0 = 1.0
t_substrate = 1.5
x_pillars0 = x_sub0 + t_substrate
x_pillars1 = x_pillars0 + pillar_height
x_focus = x_pillars1 + focal_length
x_end = x_focus + 3.0
transverse = aperture + 5.0

sim_size = np.array([x_end, transverse, transverse])
nx, ny, nz = (int(v) for v in (sim_size / pixel_size).astype(int))
cy = cz = transverse / 2

print(f"Domain {sim_size} μm = {nx}x{ny}x{nz} ({nx*ny*nz/1e6:.1f} Mvoxel); "
      f"aperture {aperture} μm, f {focal_length} μm, NA {NA:.2f}")
print(f"Pillars: n={n_pillar}, pitch {lattice_pitch} μm (~{lattice_pitch/pixel_size:.1f} px), "
      f"height {pillar_height} μm")


# --- Phase 1: calibrate the diameter → phase meta-atom library ----------------
def calibrate_library():
    """Solve periodic unit cells over a diameter sweep; return (diameters, phases).

    Local-periodic approximation: one pillar in a 1-pitch cell, periodic in Y and Z
    (an infinite array of identical pillars). Phase is measured on the transmitted
    plane relative to the no-pillar (d=0) reference, then unwrapped.
    """
    nuc = int(round(lattice_pitch / pixel_size))  # unit-cell transverse size (Y, Z)
    cuc = (nuc * pixel_size) / 2
    xs = 0.3
    xsub = 0.6
    tsub = 1.0
    xp0 = xsub + tsub
    xmeas = xp0 + pillar_height + 1.0
    xend = xmeas + 1.0
    nxu = int(xend / pixel_size)
    ixs = int(xs / pixel_size)
    ixm = int(xmeas / pixel_size)
    xg = np.arange(nxu) * pixel_size
    yg = np.arange(nuc) * pixel_size

    # Unique grid-snapped diameters (snapping linspace creates duplicates; dedupe so
    # the sweep isn't wasted and the phase→diameter interpolation stays monotonic).
    diameters = np.unique(np.round(np.linspace(0.0, lattice_pitch * 0.96, 16) / pixel_size)
                          * pixel_size)
    # y-polarized plane wave, uniform across the periodic cell. Axis order (pol,x,y,z).
    src = np.ones((1, 1, nuc, nuc), dtype=np.complex64)
    pos = (1, ixs, 0, 0)

    print(f"\nCalibrating {len(diameters)} diameters in a {nxu}x{nuc}x{nuc} unit cell "
          f"(periodic Y,Z)...")
    fields = []
    t0 = time()
    for d in diameters:
        nmap = np.full((nxu, nuc, nuc), n_bg, dtype=np.complex64)
        nmap[(xg >= xsub) & (xg < xp0), :, :] = n_substrate
        if d >= pixel_size:
            Yc, Zc = np.meshgrid(yg - cuc, yg - cuc, indexing="ij")
            disc = (Yc ** 2 + Zc ** 2) < (d / 2) ** 2
            ix0, ix1 = int(xp0 / pixel_size), int((xp0 + pillar_height) / pixel_size) + 1
            nmap[ix0:ix1] = np.where(disc[None, :, :], n_pillar, nmap[ix0:ix1])
        u, iters, res = simulate(
            permittivity=(nmap ** 2).astype(np.complex64),
            sources=[(src, pos)], wavelength=wavelength, pixel_size=pixel_size,
            boundary_width=1.0, periodic=(False, True, True), use_gpu=True,
        )
        u = np.asarray(u)
        ey = u[1]
        if ey.ndim == 4:
            ey = ey[..., 0] if ey.shape[-1] == 1 else ey
        val = ey[ixm].mean()  # complex transmitted field, averaged over the cell
        fields.append(val)
        print(f"  d={d:.3f} μm  |T|={abs(val):.3e}  ph={np.angle(val):+.2f}  "
              f"(iters {iters}, res {np.asarray(res).ravel()[-1]:.1e})")
    print(f"Calibration sweep {time()-t0:.1f} s")

    fields = np.array(fields)
    phases = np.unwrap(np.angle(fields / fields[0]))
    # Make phase increase with diameter (sign of the imprinted delay is conventional).
    if phases[-1] < phases[0]:
        phases = -phases
    return diameters, phases - phases.min()


diameters_cal, phases_cal = calibrate_library()
span = phases_cal.max() - phases_cal.min()
print(f"\nLibrary phase span = {span/(2*np.pi):.2f}·2π over d = 0…{diameters_cal[-1]:.3f} μm")
if not np.all(np.isfinite(phases_cal)) or span < 1.9 * np.pi:
    print("ABORT: calibration did not produce a clean ≥2π library "
          "(NaNs or insufficient span). Increase pillar_height or check convergence; "
          "not launching the expensive full solve.", file=sys.stderr)
    sys.exit(1)


# --- Phase 2: build the pillar forest from the measured library ---------------
def diameter_for_phase(target):
    """Invert the measured library: target wrapped phase [0,2π) → pillar diameter."""
    # phases_cal is monotonic increasing in diameter and spans ≥2π.
    return np.interp(np.clip(target, 0, phases_cal.max()), phases_cal, diameters_cal)


def build_pillar_forest():
    n = np.full((nx, ny, nz), n_bg, dtype=np.complex64)
    x = np.arange(nx) * pixel_size
    y = np.arange(ny) * pixel_size
    z = np.arange(nz) * pixel_size
    n[(x >= x_sub0) & (x < x_pillars0), :, :] = n_substrate

    n_lat = int(aperture / lattice_pitch)
    coords = (np.arange(n_lat) - (n_lat - 1) / 2) * lattice_pitch
    R_ap = aperture / 2
    ix0, ix1 = int(x_pillars0 / pixel_size), int(x_pillars1 / pixel_size)

    placed = 0
    for yc0 in coords:
        for zc0 in coords:
            r = np.hypot(yc0, zc0)
            if r > R_ap:
                continue
            phase = np.mod(focusing_sign * k0 * (np.sqrt(r ** 2 + focal_length ** 2) - focal_length),
                           2 * np.pi)
            d = float(diameter_for_phase(phase))
            if d < 0.6 * pixel_size:
                continue
            yc, zc = cy + yc0, cz + zc0
            rp = d / 2
            iy0, iy1 = max(int((yc - rp) / pixel_size), 0), min(int((yc + rp) / pixel_size) + 1, ny)
            iz0, iz1 = max(int((zc - rp) / pixel_size), 0), min(int((zc + rp) / pixel_size) + 1, nz)
            if iy0 >= iy1 or iz0 >= iz1:
                continue
            dy = (y[iy0:iy1] - yc)[:, None]
            dz = (z[iz0:iz1] - zc)[None, :]
            disc = dy ** 2 + dz ** 2 < rp ** 2
            block = n[ix0:ix1, iy0:iy1, iz0:iz1]
            block[:, disc] = n_pillar
            n[ix0:ix1, iy0:iy1, iz0:iz1] = block
            placed += 1
    print(f"Placed {placed} pillars across the {aperture} μm aperture")
    return (n ** 2).astype(np.complex64)


permittivity = build_pillar_forest()


# --- Phase 3: source + full solve ---------------------------------------------
src_y = tukey(ny, 0.3).astype(np.complex64)
src_z = tukey(nz, 0.3).astype(np.complex64)
sheet = (src_y[:, None] * src_z[None, :])
source_values = sheet[None, None, :, :].astype(np.complex64)  # (1, 1, ny, nz)
source_position = (1, int(x_src / pixel_size), 0, 0)          # y-polarized

print("\nRunning full simulation...")
start = time()
u, iterations, residual_norm = simulate(
    permittivity=permittivity,
    sources=[(source_values, source_position)],
    wavelength=wavelength,
    pixel_size=pixel_size,
    boundary_width=1.5,
    periodic=(False, False, False),
    use_gpu=True,
)
print(f"Time {time()-start:.1f} s; Iterations {iterations}; Residual {residual_norm:.2e}")

u = np.asarray(u)
intensity = (np.abs(u) ** 2).sum(axis=0)
amp = np.sqrt(intensity)

try:
    signal.signal(signal.SIGINT, signal.default_int_handler)
except (ValueError, TypeError):
    pass

ds = 2
np.save(f"{OUTPUT_DIR}/metalens_nanopillar_volume.npy", amp[::ds, ::ds, ::ds].astype(np.float32))
np.save(f"{OUTPUT_DIR}/metalens_nanopillar_index.npy",
        np.sqrt(permittivity.real)[::ds, ::ds, ::ds].astype(np.float32))

iy_c, iz_c = ny // 2, nz // 2
axis_band = intensity[:, iy_c - 1:iy_c + 2, iz_c - 1:iz_c + 2].mean(axis=(1, 2))
ix_air0 = int((x_pillars1 + 0.5) / pixel_size)
ix_focus = ix_air0 + int(np.argmax(axis_band[ix_air0:]))
x_focus_meas = ix_focus * pixel_size
print(f"\nMeasured focus at x = {x_focus_meas:.2f} μm (design {x_focus:.2f} μm)")

# --- Figure 1: focal cone -----------------------------------------------------
sl = amp[:, :, iz_c]
vmax = sl.max(); vmin = vmax * 1e-2
extent = [0, nx * pixel_size, ny * pixel_size, 0]
fig, ax = plt.subplots(figsize=(11, 7))
im = ax.imshow(sl.T, cmap="inferno", extent=extent,
               norm=colors.LogNorm(vmin=vmin, vmax=vmax), aspect="auto")
ax.axvline(x_pillars0, color="cyan", lw=0.8, ls="--", alpha=0.7)
ax.axvline(x_pillars1, color="cyan", lw=0.8, ls="--", alpha=0.7)
ax.plot(x_focus_meas, cy, "+", color="white", ms=12, mew=2)
ax.set_xlim(x_sub0, nx * pixel_size)
ax.set_xlabel(r"$x~(\mu m)$ — propagation / depth")
ax.set_ylabel(r"$y~(\mu m)$ — aperture")
ax.set_title(f"Nanopillar metalens focal cone (aperture {aperture} μm, f = {focal_length} μm, NA = {NA:.2f})\n"
             "sub-wavelength TiO₂ pillar forest (cyan), calibrated diameter→phase library")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.set_title(r"$|E|$")
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/metalens_nanopillar_focal_cone.png", dpi=300, bbox_inches="tight")

# --- Figure 2: Airy spot + FWHM -----------------------------------------------
focal_plane = intensity[ix_focus, :, :]
y = np.arange(ny) * pixel_size
cut = intensity[ix_focus, :, iz_c]; cut_n = cut / cut.max()
iy_pk = int(np.argmax(cut_n))
left = np.where(cut_n[:iy_pk] <= 0.5)[0]
right = np.where(cut_n[iy_pk:] <= 0.5)[0]
fwhm = (right[0] + iy_pk - left[-1]) * pixel_size if (len(left) and len(right)) else float("nan")

extent_t = [0, ny * pixel_size, nz * pixel_size, 0]
fig2, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
imf = axa.imshow(focal_plane.T, cmap="inferno", extent=extent_t, aspect="equal")
axa.set_title(f"Focal-plane intensity (Airy spot)\nx = {x_focus_meas:.1f} μm")
axa.set_xlabel(r"$y~(\mu m)$"); axa.set_ylabel(r"$z~(\mu m)$")
axa.set_xlim(cy - 3, cy + 3); axa.set_ylim(cz + 3, cz - 3)
fig2.colorbar(imf, ax=axa, fraction=0.046, pad=0.04).ax.set_title(r"$|E|^2$")
axb.plot(y - cy, cut_n, "-", color="tab:orange", lw=1.5)
axb.axhline(0.5, color="grey", ls=":", lw=0.8)
axb.set_xlim(-3, 3)
axb.set_xlabel(r"$y - y_0~(\mu m)$"); axb.set_ylabel(r"normalized $|E|^2$")
axb.set_title(f"Focal cut vs diffraction limit\n"
              f"measured {fwhm*1000:.0f} nm  /  0.51 λ/NA = {dl_fwhm*1000:.0f} nm")
axb.grid(True, alpha=0.3)
fig2.savefig(f"{OUTPUT_DIR}/metalens_nanopillar_airy_spot.png", dpi=300, bbox_inches="tight")

print(f"Focal-spot FWHM: measured {fwhm*1000:.0f} nm vs diffraction limit {dl_fwhm*1000:.0f} nm")
print(f"Saved {OUTPUT_DIR}/metalens_nanopillar_focal_cone.png and _airy_spot.png")
plt.show()
