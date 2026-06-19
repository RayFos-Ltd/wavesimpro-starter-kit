"""
Maxwell 2D multimode-interference (MMI) 1x2 splitter
====================================================
A 1x2 MMI power splitter: a single-mode access waveguide feeds a short, wide
*multimode* section; the input excites several lateral modes that beat against
each other, and at the right length the field reforms into two identical copies
of the input -- a 50/50 split -- which are caught by two output waveguides. This
"self-imaging" is the workhorse splitter/combiner of silicon photonics.

Top-down view (the simulation plane):
  * axis 0 = x = propagation along the chip,
  * axis 1 = y = lateral (across the waveguides),
  * axis 2 = z = thin / periodic (an effective-index 2D reduction of the slab).

Effective-index model: the 220 nm Si slab is collapsed to a 2D problem by using
its slab effective index (~2.84) as the in-plane core index in an SiO2 cladding.

Why a full-wave solver: self-imaging is a coherent interference effect of many
guided modes; the split ratio and length depend on the modal phase relations, a
near-field interference pattern with no ray-optics description. MBS resolves the
whole multimode beat over the device length with no numerical dispersion to
corrupt the modal phases.

The device is *designed by the mode solver itself*: ``solve_waveguide_modes``
(the svmodes finite-difference solver) gives the multimode-section modes and
propagation constants; propagating the input field through them (a modal
expansion) locates the two-image plane and the output positions. The full-wave
``simulate`` run then validates that design. TE polarization (E_z out of plane).
"""

import os
import signal
from time import time

import numpy as np

from wavesimpro import simulate
from wsutils import (
    index_canvas,
    add_waveguide,
    permittivity_2d,
    solve_waveguide_modes,
    solve_waveguide_mode,
    mode_source,
    OUTPUT_DIR,
)

# --- Parameters (edit these to rescale) --------------------------------------
wavelength = 1.55  # design wavelength in micrometer (μm)
pixel_size = 0.04  # grid spacing in micrometer (μm)

n_core = 2.835  # in-plane core index ~ slab effective index of a 220 nm SOI core
n_clad = 1.444  # SiO2 cladding

W_mmi = 3.0  # multimode section width (μm)
w_wg = 0.5  # single-mode access/output waveguide width (μm)

sim_y = W_mmi + 4.0  # lateral domain: MMI width + cladding margin (μm)
yc = sim_y / 2  # transverse center

x_source = 2.5  # mode launch position (μm), clear of the boundary
mmi_start = 4.0  # x where the multimode section begins (μm)
boundary_width = 1.5  # absorbing boundary width (μm)

# --- Design the MMI with the integrated mode solver --------------------------
# Build the lateral index profile of the multimode section and of the access
# waveguide, both sampled on the full transverse grid.
ny = int(sim_y / pixel_size)
y = np.arange(ny) * pixel_size

mmi_profile = np.full(ny, n_clad, dtype=np.complex64)
mmi_profile[np.abs(y - yc) < W_mmi / 2] = n_core

access_profile = np.full(ny, n_clad, dtype=np.complex64)
access_profile[np.abs(y - yc) < w_wg / 2] = n_core

# Multimode-section modes (keep only the guided ones) and the input field.
phi_modes, neff_modes = solve_waveguide_modes(mmi_profile, pixel_size, wavelength, nmodes=10, field="scalar", guess=n_core)
guided = neff_modes.real > n_clad
phi_modes = phi_modes[:, guided].real
beta = 2 * np.pi * neff_modes[guided].real / wavelength
e_in, _ = solve_waveguide_mode(access_profile, pixel_size, wavelength, field="scalar", guess=n_core)
e_in = e_in.real
print(f"MMI: {phi_modes.shape[1]} guided modes; beat length L_pi = {np.pi / (beta[0] - beta[1]):.2f} um")

# Modal expansion: coefficients = overlap of the input with each mode.
coeff = (e_in @ phi_modes) / np.sum(phi_modes**2, axis=0)

# Propagate the modal field and find the first strong two-fold image plane.
x_scan = np.linspace(0.0, 1.6 * np.pi / (beta[0] - beta[1]), 600)
field_scan = (coeff * np.exp(-1j * beta[None, :] * x_scan[:, None])) @ phi_modes.T
I_scan = np.abs(field_scan) ** 2


def _band(y_target, half=0.3):
    return np.abs(y - y_target) < half


# Score: reward two side lobes at +/- W/4, penalize a central lobe; ignore x near input.
score = np.array([
    I_scan[i, _band(yc - W_mmi / 4)].sum() + I_scan[i, _band(yc + W_mmi / 4)].sum() - 2 * I_scan[i, _band(yc)].sum()
    for i in range(len(x_scan))
])
sel = x_scan > 3.0
L_mmi = float(x_scan[sel][np.argmax(score[sel])])

# Two output lobe positions at that plane (from the modal field).
i_img = int(np.argmin(np.abs(x_scan - L_mmi)))
prof = I_scan[i_img]
y_up = y[y < yc][np.argmax(prof[y < yc])]
y_lo = y[y >= yc][np.argmax(prof[y >= yc])]
print(f"Designed MMI length L_mmi = {L_mmi:.2f} um; output centers y = {y_up:.2f}, {y_lo:.2f} um")

mmi_end = mmi_start + L_mmi

# --- Build the geometry ------------------------------------------------------
sim_size = np.array([mmi_end + 4.0, sim_y])  # leave room for the output waveguides
n = index_canvas(sim_size, pixel_size, background=n_clad)
add_waveguide(n, pixel_size, (0, mmi_start), yc, w_wg, n_core)  # input access waveguide
add_waveguide(n, pixel_size, (mmi_start, mmi_end), yc, W_mmi, n_core)  # multimode section
add_waveguide(n, pixel_size, (mmi_end, sim_size[0]), y_up, w_wg, n_core)  # output 1
add_waveguide(n, pixel_size, (mmi_end, sim_size[0]), y_lo, w_wg, n_core)  # output 2

permittivity = permittivity_2d(n)
nx = permittivity.shape[0]

# --- Source: fundamental mode of the access waveguide ------------------------
source_values, source_position = mode_source(
    e_in.astype(np.complex64),
    origin="topleft",
    position=[2, x_source, 0.0, 0.0],  # z-polarized (TE), full-height profile from y=0
    source_plane="y",
    pixel_size=pixel_size,
)

# --- Run the simulation ------------------------------------------------------
print("Running simulation...")
start = time()
u, iterations, residual_norm = simulate(
    permittivity=permittivity,
    sources=[(source_values, source_position)],
    wavelength=wavelength,
    pixel_size=pixel_size,
    boundary_width=boundary_width,
    periodic=(False, False, True),  # 2D: uniform out of plane
    use_gpu=True,
)
print(f"Time {time() - start:.1f} s; Iterations {iterations}; Residual {residual_norm:.2e}")

# E_z field as a 2D (nx, ny) slice
ez = u[2]
if ez.ndim == 3:
    ez = ez[:, :, ez.shape[2] // 2]

# Restore a sane SIGINT handler before plotting.
try:
    signal.signal(signal.SIGINT, signal.default_int_handler)
except (ValueError, TypeError):
    pass

# The money shot: the multimode interference "carpet" inside the MMI section and
# the clean 50/50 split into the two output waveguides.
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

intensity = np.abs(ez) ** 2
intensity /= intensity.max()
extent = [0, sim_size[0], sim_size[1], 0]

plt.figure(figsize=(11, 4.5))
plt.imshow(intensity.T, cmap="inferno", extent=extent, norm=LogNorm(vmin=1e-3, vmax=1.0), aspect="equal")
cbar = plt.colorbar(fraction=0.025, pad=0.02)
cbar.ax.set_title(r"$|E_z|^2$")
plt.axvline(mmi_start, color="cyan", lw=0.5, alpha=0.4)
plt.axvline(mmi_end, color="cyan", lw=0.5, alpha=0.4)
plt.xlabel(r"$x~(\mu m)$ -- propagation")
plt.ylabel(r"$y~(\mu m)$ -- lateral")
plt.title(f"1x2 MMI splitter: self-imaging 50/50 split (W={W_mmi} um, L={L_mmi:.1f} um)")
plt.tight_layout()

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.savefig(f"{OUTPUT_DIR}/maxwell_mmi_splitter.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{OUTPUT_DIR}/maxwell_mmi_splitter.pdf", bbox_inches="tight")
plt.show()
plt.close("all")
