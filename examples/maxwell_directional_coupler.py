"""
Maxwell 2D directional coupler
==============================
Two single-mode waveguides run side by side with a sub-wavelength gap. The
evanescent tail of the mode in one guide reaches into the other, so power leaks
across and sloshes back and forth along the device -- the basis of every
directional-coupler splitter, switch, and ring-bus coupler in integrated optics.

Top-down view (the simulation plane):
  * axis 0 = x = propagation along the chip,
  * axis 1 = y = lateral (across the two guides),
  * axis 2 = z = thin / periodic (an effective-index 2D reduction of the slab).

The coupling is set by the beat between the two *supermodes* of the pair (the
even/symmetric and odd/antisymmetric combinations): light launched in one guide
transfers completely to the other over the coupling length
``L_c = pi / (beta_sym - beta_antisym)``, then returns, with period ``2 L_c``.

Why a full-wave solver: evanescent coupling is a near-field overlap of
exponential mode tails across the gap -- there is no ray-optics picture of light
"jumping" between guides. The transfer length depends on the supermode phase
difference, which MBS resolves with no numerical dispersion over the long, thin
device. (FDTD pain point: dispersion error accumulates over the many-L_c length.)

``solve_waveguide_modes`` (the svmodes solver) computes the two supermodes and
hence L_c, which sets the device length; the full-wave ``simulate`` run then
shows the actual power exchange. TE polarization (E_z out of plane).
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

w_wg = 0.5  # single-mode waveguide width (μm)
gap = 0.20  # edge-to-edge gap between the two guides (μm) -> strong coupling
n_coupling_lengths = 3.0  # device length in units of L_c (3 -> 1.5 full beat periods)

sim_y = 5.0  # lateral domain (μm)
yc = sim_y / 2  # transverse center
sep = (w_wg + gap) / 2  # center-to-center half-separation
y_top = yc - sep  # upper guide center (the launch guide)
y_bot = yc + sep  # lower guide center

x_source = 2.5  # mode launch position (μm), clear of the boundary
boundary_width = 1.5  # absorbing boundary width (μm)

# --- Coupling length from the two supermodes ---------------------------------
ny = int(sim_y / pixel_size)
y = np.arange(ny) * pixel_size


def lateral_index(cores):
    """Build a 1D lateral refractive-index profile using the *same* pixel
    rasterization as the simulated geometry (a 1-px-wide painted strip), so the
    mode analysis and the full-wave run see an identical cross-section. With a
    sub-wavelength gap the coupling is exponentially sensitive to discretization,
    so this consistency is what makes the predicted L_c match the observed one."""
    strip = index_canvas([pixel_size, sim_y], pixel_size, background=n_clad)  # (1, ny)
    for (y_center, width) in cores:
        add_waveguide(strip, pixel_size, (0, pixel_size), y_center, width, n_core)
    return strip[0]


pair_profile = lateral_index([(y_top, w_wg), (y_bot, w_wg)])

_, neff_pair = solve_waveguide_modes(pair_profile, pixel_size, wavelength, nmodes=2, field="scalar", guess=n_core)
beta_sym, beta_anti = 2 * np.pi * neff_pair[0].real / wavelength, 2 * np.pi * neff_pair[1].real / wavelength
L_c = np.pi / (beta_sym - beta_anti)
coupling_length = n_coupling_lengths * L_c
print(f"Supermodes: neff_sym={neff_pair[0].real:.4f}, neff_anti={neff_pair[1].real:.4f}")
print(f"Coupling length L_c = {L_c:.2f} um; device coupled region = {coupling_length:.2f} um")

# --- Build the geometry: two parallel guides across the whole domain ----------
sim_size = np.array([x_source + coupling_length + 3.0, sim_y])
n = index_canvas(sim_size, pixel_size, background=n_clad)
add_waveguide(n, pixel_size, (0, sim_size[0]), y_top, w_wg, n_core)
add_waveguide(n, pixel_size, (0, sim_size[0]), y_bot, w_wg, n_core)
permittivity = permittivity_2d(n)
nx = permittivity.shape[0]

# --- Source: fundamental mode of a single (isolated) guide, in the top guide --
single_profile = lateral_index([(y_top, w_wg)])
e_in, _ = solve_waveguide_mode(single_profile, pixel_size, wavelength, field="scalar", guess=n_core)

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

# Report the power in each guide along x (a clean illustration of the exchange).
intensity = np.abs(ez) ** 2
x = np.arange(nx) * pixel_size
top_band = np.abs(y - y_top) < (w_wg / 2 + gap / 2)
bot_band = np.abs(y - y_bot) < (w_wg / 2 + gap / 2)
p_top = intensity[:, top_band].sum(axis=1)
p_bot = intensity[:, bot_band].sum(axis=1)

# Restore a sane SIGINT handler before plotting.
try:
    signal.signal(signal.SIGINT, signal.default_int_handler)
except (ValueError, TypeError):
    pass

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

extent = [0, sim_size[0], sim_size[1], 0]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), height_ratios=[2, 1], sharex=True)

im = ax1.imshow(intensity.T / intensity.max(), cmap="inferno", extent=extent,
                norm=LogNorm(vmin=1e-3, vmax=1.0), aspect="auto")
ax1.set_ylabel(r"$y~(\mu m)$ -- lateral")
ax1.set_title(f"Directional coupler: evanescent power exchange (gap={gap} um, $L_c$={L_c:.1f} um)")
fig.colorbar(im, ax=ax1, fraction=0.025, pad=0.02).ax.set_title(r"$|E_z|^2$")

# Power in each guide vs propagation distance (normalized to the launched total)
norm = (p_top + p_bot)[int(x_source / pixel_size) + 5]
ax2.plot(x, p_top / norm, label="top guide", color="tab:orange")
ax2.plot(x, p_bot / norm, label="bottom guide", color="tab:cyan")
ax2.set_xlabel(r"$x~(\mu m)$ -- propagation")
ax2.set_ylabel("relative power")
ax2.set_xlim(0, sim_size[0])
ax2.legend(loc="upper right", fontsize=8)
ax2.grid(alpha=0.3)
plt.tight_layout()

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.savefig(f"{OUTPUT_DIR}/maxwell_directional_coupler.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{OUTPUT_DIR}/maxwell_directional_coupler.pdf", bbox_inches="tight")
plt.show()
plt.close("all")
