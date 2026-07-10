"""
Maxwell 3D multimode-interference (MMI) 1x2 splitter
====================================================
A 1x2 MMI power splitter on a silicon-on-insulator (SOI) platform: a single-mode
access waveguide feeds a short, wide *multimode* section; the input excites
several modes that beat against each other, and at the right length the field
reforms into two identical copies of the input -- a 50/50 split -- caught by two
output waveguides. This "self-imaging" is the workhorse splitter/combiner of
silicon photonics.

This is a genuine 3D structure (no effective-index reduction): a 220 nm-thick
silicon core embedded in SiO2, with real material indices. The simulation axes are
  * x = propagation along the chip,
  * y = lateral (across the waveguides, the guide WIDTH direction),
  * z = vertical (the 220 nm layer-stack / core-thickness direction).
The waveguide cross-section transverse to propagation is the real 2D (y, z) plane,
which is exactly the input the finite-difference mode solver ``wsutils.svmodes``
expects -- so the device is *designed by the mode solver itself*.

Why a full-wave solver: self-imaging is a coherent interference effect of many
guided modes; the split ratio and length depend on the modal phase relations, a
near-field interference pattern with no ray-optics description. The Modified Born
Series solver resolves the whole multimode beat over the device length with no
numerical dispersion to corrupt the modal phases.

Design flow: ``svmodes`` gives the multimode-section modes and propagation
constants on the (y, z) cross-section; a modal expansion of the launched access
mode locates the two-image plane and the output positions. The full-wave
``simulate`` run then validates that design. Quasi-TE polarization (dominant E
along the guide width, i.e. y).
"""

import os
import signal
from time import time

import numpy as np

from wavesimpro import simulate, create_monitor
from wsutils import index_canvas, add_waveguide, svmodes, mode_source, OUTPUT_DIR

# --- Parameters (edit these to rescale) --------------------------------------
wavelength = 1.55  # design wavelength in micrometer (μm)
pixel_size = 0.025  # grid spacing (μm); ~9 px across the 220 nm core thickness

n_si = 3.476   # silicon core
n_clad = 1.444  # SiO2 cladding (buried oxide below + oxide over-cladding)

t_core = 0.22  # Si core thickness, vertical z (μm)
W_mmi = 3.0    # multimode-section width, lateral y (μm)
w_wg = 0.45    # single-mode access/output waveguide width (μm)

# Transverse window (y lateral, z vertical), core centered in both.
sim_y = W_mmi + 3.0  # lateral domain: MMI width + cladding/boundary margin (μm)
sim_z = t_core + 2.4  # vertical domain: core + oxide margin above/below (μm)
yc = sim_y / 2
zc = sim_z / 2

x_source = 2.5       # mode launch position (μm), clear of the boundary
mmi_start = 4.0      # x where the multimode section begins (μm)
boundary_width = 6.0  # absorbing boundary width (μm) ≈4.0λ (MBS sizing rule), all six faces

ny = int(round(sim_y / pixel_size))
nz = int(round(sim_z / pixel_size))
# core z-band (vertical extent of the silicon layer)
kz0 = int(round((zc - t_core / 2) / pixel_size))
kz1 = int(round((zc + t_core / 2) / pixel_size))


def cross_section(core_width):
    """Permittivity n^2 of a (y, z) cross-section: a ``core_width`` x ``t_core``
    Si core centered in the window, embedded in SiO2. Returns shape (nz, ny) --
    svmodes' (vertical, horizontal) = (z, y) convention."""
    n = np.full((nz, ny), n_clad, dtype=np.complex64)
    jy0 = int(round((yc - core_width / 2) / pixel_size))
    jy1 = int(round((yc + core_width / 2) / pixel_size))
    n[kz0:kz1, jy0:jy1] = n_si
    return (n ** 2)


# --- Design the MMI with the integrated mode solver --------------------------
# Access-waveguide fundamental mode = the launched field. Quasi-TE -> field 'EX'
# (E along the horizontal/y = guide width); see wsutils.svmodes conventions.
# svmodes returns (neff, phi); guess = core index n_si.
neff_in, phi_in = svmodes(wavelength, n_si, 1, pixel_size, pixel_size, cross_section(w_wg), "0000", "EX")
e_in = phi_in.real  # (nz, ny)

# Multimode-section modes: solve a generous set, keep the guided ones (neff > clad).
n_solve = 14
neff_mmi, phi_mmi = svmodes(wavelength, n_si, n_solve, pixel_size, pixel_size, cross_section(W_mmi), "0000", "EX")
guided = neff_mmi.real > n_clad + 1e-3
neff_mmi = neff_mmi[guided]
phi_mmi = phi_mmi[:, :, guided].real  # (nz, ny, M)
beta = 2 * np.pi * neff_mmi.real / wavelength
M = phi_mmi.shape[2]
print(f"Access neff = {neff_in.real:.4f}; MMI guided modes = {M}")
print(f"MMI beat length L_pi = pi/(b0-b1) = {np.pi / (beta[0] - beta[1]):.2f} um")

# Modal expansion: coefficients = 2D overlap of the input with each MMI mode.
norm_m = np.sum(phi_mmi ** 2, axis=(0, 1))  # (M,)
coeff = np.sum(e_in[:, :, None] * phi_mmi, axis=(0, 1)) / norm_m  # (M,)

# Propagate the modal field; evaluate intensity on the mid-core-height z-plane and
# scan along x to find the first strong two-fold image (two lobes at +/- W/4).
z_mid = (kz0 + kz1) // 2
y = np.arange(ny) * pixel_size
x_scan = np.linspace(0.0, 1.8 * np.pi / (beta[0] - beta[1]), 700)
# field(x, y) at mid height = sum_m coeff_m phi_m(z_mid, y) exp(-i beta_m x)
phi_mid = phi_mmi[z_mid, :, :]  # (ny, M)
field_scan = (coeff * np.exp(-1j * beta[None, :] * x_scan[:, None])) @ phi_mid.T  # (Nx, ny)
I_scan = np.abs(field_scan) ** 2


def _band(y_target, half=0.3):
    return np.abs(y - y_target) < half


score = np.array([
    I_scan[i, _band(yc - W_mmi / 4)].sum() + I_scan[i, _band(yc + W_mmi / 4)].sum() - 2 * I_scan[i, _band(yc)].sum()
    for i in range(len(x_scan))
])
sel = x_scan > 3.0
L_mmi = float(x_scan[sel][np.argmax(score[sel])])

i_img = int(np.argmin(np.abs(x_scan - L_mmi)))
prof = I_scan[i_img]
y_up = y[y < yc][np.argmax(prof[y < yc])]
y_lo = y[y >= yc][np.argmax(prof[y >= yc])]
mmi_end = mmi_start + L_mmi
print(f"Designed MMI length L_mmi = {L_mmi:.2f} um; output centers y = {y_up:.2f}, {y_lo:.2f} um")

# --- Build the 3D geometry ---------------------------------------------------
# In-plane (top-down x-y) index pattern at core height, then extrude into the
# 220 nm z core-band; everything else is SiO2.
sim_x = mmi_end + 4.0  # room for the output waveguides
nxy = index_canvas([sim_x, sim_y], pixel_size, background=n_clad)  # (nx, ny)
add_waveguide(nxy, pixel_size, (0, mmi_start), yc, w_wg, n_si)        # input access waveguide
add_waveguide(nxy, pixel_size, (mmi_start, mmi_end), yc, W_mmi, n_si)  # multimode section
add_waveguide(nxy, pixel_size, (mmi_end, sim_x), y_up, w_wg, n_si)     # output 1
add_waveguide(nxy, pixel_size, (mmi_end, sim_x), y_lo, w_wg, n_si)     # output 2
nx = nxy.shape[0]

permittivity = np.full((nx, ny, nz), n_clad ** 2, dtype=np.complex64)
permittivity[:, :, kz0:kz1] = (nxy ** 2)[:, :, None]
mvox = nx * ny * nz / 1e6
print(f"Grid: {nx} x {ny} x {nz} = {mvox:.1f} Mvoxel  (px={pixel_size} um)")

# --- Source: launched access mode over the (y, z) plane ----------------------
# svmodes phi is (nz, ny); mode_source's 'yz' plane wants (n_y, n_z) -> transpose.
source_values, source_position = mode_source(
    e_in.T.astype(np.complex64),       # (ny, nz)
    origin="topleft",
    position=[1, x_source, 0.0, 0.0],  # y-polarized (quasi-TE); profile fills the y,z window
    source_plane="yz",
    pixel_size=pixel_size,
)


# --- Monitor (recommended): return just the x-y cut-plane this figure plots ---
# A monitor crops the result to this plane server-side, so only a thin slice is
# transferred instead of the full volume -- much faster, with identical values on
# the plotted plane. Keep it on; set USE_MONITOR = False only to fetch the whole
# 3D volume. The monitor is a thin slab at mid core height; voxel index i on an
# n-voxel axis sits at (i + 0.5 - n/2)*px in domain-centred world coordinates.
USE_MONITOR = True
if USE_MONITOR:
    z_world = (z_mid + 0.5 - nz / 2) * pixel_size
    monitor_list = [create_monitor(
        name="cutplane",
        center_x=0.0, center_y=0.0, center_z=z_world,
        size_x=nx * pixel_size, size_y=ny * pixel_size, size_z=3 * pixel_size,
    )]
else:
    monitor_list = None

# --- Run the simulation ------------------------------------------------------
print("Running simulation...")
start = time()
u, iterations, residual_norm = simulate(
    permittivity=permittivity,
    sources=[(source_values, source_position)],
    wavelength=wavelength,
    pixel_size=pixel_size,
    boundary_width=boundary_width,
    periodic=(False, False, False),  # genuine 3D: absorbing on all six faces
    use_gpu=True,
    monitors=monitor_list,
)
print(f"Time {time() - start:.1f} s; Iterations {iterations}; Residual {residual_norm:.2e}")

if USE_MONITOR:
    u = u[0]   # monitor run returns a list of fields; take the only one

# From here on, u is the field volume in both cases. Visualization cut-plane:
# top-down slice at mid core height (the mid z index).
ey = u[1]  # dominant (quasi-TE) component
field_top = ey[:, :, ey.shape[2] // 2]

try:
    signal.signal(signal.SIGINT, signal.default_int_handler)
except (ValueError, TypeError):
    pass

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

intensity = np.abs(field_top) ** 2
intensity /= intensity.max()
extent = [0, sim_x, sim_y, 0]

plt.figure(figsize=(11, 4.5))
plt.imshow(intensity.T, cmap="inferno", extent=extent, norm=LogNorm(vmin=1e-3, vmax=1.0), aspect="equal")
cbar = plt.colorbar(fraction=0.025, pad=0.02)
cbar.ax.set_title(r"$|E_y|^2$")
plt.axvline(mmi_start, color="cyan", lw=0.5, alpha=0.4)
plt.axvline(mmi_end, color="cyan", lw=0.5, alpha=0.4)
plt.xlabel(r"$x~(\mu m)$ -- propagation")
plt.ylabel(r"$y~(\mu m)$ -- lateral")
plt.title(f"3D 1x2 MMI splitter: self-imaging 50/50 split (W={W_mmi} um, L={L_mmi:.1f} um, mid-core z)")
plt.tight_layout()

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.savefig(f"{OUTPUT_DIR}/maxwell_mmi_splitter.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{OUTPUT_DIR}/maxwell_mmi_splitter.pdf", bbox_inches="tight")
plt.show()
plt.close("all")
