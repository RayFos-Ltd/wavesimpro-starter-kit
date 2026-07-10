"""
Maxwell 3D directional coupler
==============================
Two single-mode silicon waveguides run side by side with a sub-wavelength gap.
The evanescent tail of the mode in one guide reaches into the other, so power
leaks across and sloshes back and forth along the device -- the basis of every
directional-coupler splitter, switch, and ring-bus coupler in integrated optics.

This is a genuine 3D structure: two 450 x 220 nm silicon cores embedded in SiO2,
with real material indices. The simulation axes are
  * x = propagation along the chip,
  * y = lateral (across the two guides, the WIDTH direction),
  * z = vertical (the 220 nm core-thickness direction).

The coupling is set by the beat between the two *supermodes* of the pair (the
even/symmetric and odd/antisymmetric combinations): light launched in one guide
transfers completely to the other over the coupling length
``L_c = pi / (beta_sym - beta_antisym)``, then returns, with period ``2 L_c``.

Why a full-wave solver: evanescent coupling is a near-field overlap of
exponential mode tails across the gap -- no ray-optics picture of light "jumping"
between guides. The transfer length depends on the supermode phase difference,
which the Modified Born Series solver resolves with no numerical dispersion over
the long, thin device. (FDTD pain point: dispersion error accumulates over many
coupling lengths.)

``svmodes`` computes the two supermodes on the real (y, z) cross-section and hence
L_c, which sets the device length; the full-wave ``simulate`` run then shows the
actual power exchange. Quasi-TE polarization (dominant E along the width, y).
"""

import os
import signal
from time import time

import numpy as np

from wavesimpro import simulate, create_monitor
from wsutils import index_canvas, add_waveguide, svmodes, mode_source, OUTPUT_DIR

# --- Parameters (edit these to rescale) --------------------------------------
wavelength = 1.55  # design wavelength in micrometer (μm)
pixel_size = 0.025  # grid spacing (μm); ~9 px across the 220 nm core, 8 px across the gap

n_si = 3.476   # silicon core
n_clad = 1.444  # SiO2 cladding

t_core = 0.22  # Si core thickness, vertical z (μm)
w_wg = 0.45    # single-mode waveguide width, lateral y (μm)
gap = 0.20     # edge-to-edge gap between the two guides (μm) -> strong coupling
n_coupling_lengths = 2.0  # device length in units of L_c (2 -> one full crossover and back)

sim_y = 4.0    # lateral domain (μm)
sim_z = t_core + 2.4  # vertical domain: core + oxide margin (μm)
yc = sim_y / 2
zc = sim_z / 2
sep = (w_wg + gap) / 2  # center-to-center half-separation
y_top = yc - sep  # upper guide center (the launch guide)
y_bot = yc + sep  # lower guide center

x_source = 2.5       # mode launch position (μm), clear of the boundary
boundary_width = 6.0  # absorbing boundary width (μm) ≈4.0λ (MBS sizing rule), all six faces

ny = int(round(sim_y / pixel_size))
nz = int(round(sim_z / pixel_size))
kz0 = int(round((zc - t_core / 2) / pixel_size))
kz1 = int(round((zc + t_core / 2) / pixel_size))


def cross_section(core_centers):
    """Permittivity n^2 of a (y, z) cross-section with a 450x220 nm Si core at each
    y in ``core_centers``, embedded in SiO2. Returns (nz, ny) = svmodes (z, y)."""
    n = np.full((nz, ny), n_clad, dtype=np.complex64)
    for y_center in core_centers:
        jy0 = int(round((y_center - w_wg / 2) / pixel_size))
        jy1 = int(round((y_center + w_wg / 2) / pixel_size))
        n[kz0:kz1, jy0:jy1] = n_si
    return (n ** 2)


# --- Coupling length from the two supermodes of the pair ---------------------
# Quasi-TE -> field 'EX' (E along horizontal/y = width). Two lowest modes of the
# coupled pair are the symmetric (even, higher neff) and antisymmetric supermodes.
neff_pair, phi_pair = svmodes(wavelength, n_si, 2, pixel_size, pixel_size, cross_section([y_top, y_bot]), "0000", "EX")
beta_sym, beta_anti = 2 * np.pi * neff_pair[0].real / wavelength, 2 * np.pi * neff_pair[1].real / wavelength
L_c = np.pi / (beta_sym - beta_anti)
coupling_length = n_coupling_lengths * L_c
print(f"Supermodes: neff_sym={neff_pair[0].real:.4f}, neff_anti={neff_pair[1].real:.4f}")
print(f"Coupling length L_c = {L_c:.2f} um; device coupled region = {coupling_length:.2f} um")

# Launched field: fundamental mode of a single (isolated) guide at the top guide.
neff_single, phi_single = svmodes(wavelength, n_si, 1, pixel_size, pixel_size, cross_section([y_top]), "0000", "EX")
e_in = phi_single.real  # (nz, ny), localized at y_top

# --- Build the 3D geometry: two parallel guides across the whole domain -------
sim_x = x_source + coupling_length + 3.0
nxy = index_canvas([sim_x, sim_y], pixel_size, background=n_clad)  # (nx, ny) at core height
add_waveguide(nxy, pixel_size, (0, sim_x), y_top, w_wg, n_si)
add_waveguide(nxy, pixel_size, (0, sim_x), y_bot, w_wg, n_si)
nx = nxy.shape[0]

permittivity = np.full((nx, ny, nz), n_clad ** 2, dtype=np.complex64)
permittivity[:, :, kz0:kz1] = (nxy ** 2)[:, :, None]
mvox = nx * ny * nz / 1e6
print(f"Grid: {nx} x {ny} x {nz} = {mvox:.1f} Mvoxel  (px={pixel_size} um)")

# --- Source: single-guide mode over the (y, z) plane, in the top guide --------
source_values, source_position = mode_source(
    e_in.T.astype(np.complex64),       # (ny, nz)
    origin="topleft",
    position=[1, x_source, 0.0, 0.0],  # y-polarized (quasi-TE); profile fills the y,z window
    source_plane="yz",
    pixel_size=pixel_size,
)

z_mid = (kz0 + kz1) // 2
y = np.arange(ny) * pixel_size

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
    monitors=monitor_list
)
print(f"Time {time() - start:.1f} s; Iterations {iterations}; Residual {residual_norm:.2e}")

if USE_MONITOR:
    u = u[0]   # monitor run returns a list of fields; take the only one we defined

# From here on, u is the field volume in both cases. Ey on the mid-core x-y plane;
# the mid z index is the core centre.
ey = u[1]
field_top = ey[:, :, ey.shape[2] // 2]

intensity = np.abs(field_top) ** 2
x = np.arange(nx) * pixel_size
top_band = np.abs(y - y_top) < (w_wg / 2 + gap / 2)
bot_band = np.abs(y - y_bot) < (w_wg / 2 + gap / 2)
p_top = intensity[:, top_band].sum(axis=1)
p_bot = intensity[:, bot_band].sum(axis=1)

try:
    signal.signal(signal.SIGINT, signal.default_int_handler)
except (ValueError, TypeError):
    pass

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

extent = [0, sim_x, sim_y, 0]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), height_ratios=[2, 1], sharex=True)

im = ax1.imshow(intensity.T / intensity.max(), cmap="inferno", extent=extent,
                norm=LogNorm(vmin=1e-3, vmax=1.0), aspect="auto")
ax1.set_ylabel(r"$y~(\mu m)$ -- lateral")
ax1.set_title(f"3D directional coupler: evanescent power exchange (gap={gap} um, $L_c$={L_c:.1f} um, mid-core z)")
fig.colorbar(im, ax=ax1, fraction=0.025, pad=0.02).ax.set_title(r"$|E_y|^2$")

norm = (p_top + p_bot)[int(x_source / pixel_size) + 5]
ax2.plot(x, p_top / norm, label="top guide (launch)", color="tab:orange")
ax2.plot(x, p_bot / norm, label="bottom guide", color="tab:cyan")
ax2.set_xlabel(r"$x~(\mu m)$ -- propagation")
ax2.set_ylabel("relative power")
ax2.set_xlim(0, sim_x)
ax2.legend(loc="upper right", fontsize=8)
ax2.grid(alpha=0.3)
plt.tight_layout()

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.savefig(f"{OUTPUT_DIR}/maxwell_directional_coupler.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{OUTPUT_DIR}/maxwell_directional_coupler.pdf", bbox_inches="tight")
plt.show()
plt.close("all")
