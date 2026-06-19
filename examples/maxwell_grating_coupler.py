"""
Maxwell 2D grating coupler (fiber-to-chip)
==========================================
A silicon-on-insulator (SOI) waveguide grating that lifts guided light off the
chip into a free-space beam aimed at a notional optical fiber. This is the
canonical photonic-integrated-circuit (PIC) input/output structure: you cannot
butt-couple a 9 μm fiber mode to a 0.5 μm waveguide, so a periodic grating
diffracts the guided mode upward (out of plane) into a beam the fiber can catch.

Side view (the simulation plane):
  * axis 0 = x = propagation along the chip (left -> right),
  * axis 1 = y = the vertical layer stack: SiO2 buried oxide (BOX) below,
    a 220 nm Si core, air cladding above,
  * axis 2 = z = thin / periodic (the wafer is uniform out of plane -> 2D).

The fundamental waveguide mode is launched from the left in the unetched Si
slab. Where it reaches the partially-etched grating, each tooth scatters a
little power; the teeth add up coherently into a single diffracted beam leaving
the surface at an angle set by the grating equation
``n_clad sin(theta) = n_eff - lambda / period`` (first diffracted order).

Why a full-wave solver (not ray tracing): the coupling is sub-wavelength tooth
scattering plus near-field overlap with the fiber mode -- diffraction, not
refraction. The FDTD pain point here is the fiber stand-off: the beam must
propagate several μm above the chip before it is "a fiber mode", inflating the
domain; MBS captures the whole near-field-to-stand-off region in one CW solve.

TE polarization (E out of plane, E_z) -- the usual silicon-photonics grating
mode. Single wavelength: coupler bandwidth would need a wavelength sweep.

Geometry is built procedurally with ``wsutils.layout`` (no GDS import), and the
launch field is the true waveguide mode from ``wsutils.solve_waveguide_mode``
(the svmodes finite-difference solver), not a Gaussian approximation.
"""

import signal
from time import time

import numpy as np

from wavesimpro import simulate
from wsutils import (
    index_canvas,
    add_rect,
    add_waveguide,
    add_grating,
    permittivity_2d,
    solve_waveguide_mode,
    mode_source,
    OUTPUT_DIR,
)

# --- Parameters (edit these to rescale) --------------------------------------
wavelength = 1.55  # design wavelength in micrometer (μm)
pixel_size = 0.02  # grid spacing in micrometer (μm); ~11 px across the 220 nm core

# Materials (refractive index n at 1.55 μm)
n_si = 3.476  # silicon core
n_box = 1.444  # SiO2 buried oxide
n_air = 1.0  # air cladding

# Layer stack (vertical y, in micrometer μm). The plot has y increasing downward,
# so we put air on top (small y) and the BOX/substrate below: the diffracted beam
# then radiates upward (toward y = 0), the way you picture it lifting off the chip.
sim_size = np.array([18.0, 8.0])  # (x, y) domain size (μm)
air_above = 4.5  # air/cladding thickness above the chip (μm) -- room for the beam
t_core = 0.22  # Si core thickness (220 nm SOI)
core_top = air_above  # air/core interface; grating radiates toward y = 0
core_bottom = core_top + t_core  # core/BOX interface (BOX fills below to the boundary)

# Grating (partial etch into the top of the core). A moderately deep etch over
# ~13 μm couples most of the guided power up within the domain (a real fiber
# grating coupler is designed to radiate over roughly the fiber mode-field size).
period = 0.60  # grating pitch Λ (μm)
duty = 0.5  # tooth fraction of the period
etch_depth = 0.13  # partial etch depth from the core top (μm)
n_teeth = 22
grating_start = 4.0  # x where the grating begins (μm)

# Source: launch the fundamental mode in the input waveguide (clear of the
# absorbing boundary, with a short lead-in before the grating).
x_source = 2.5  # x position of the mode launch (μm)
boundary_width = 2.0  # absorbing boundary width (μm)

# --- Build the geometry ------------------------------------------------------
n = index_canvas(sim_size, pixel_size, background=n_air)
add_rect(n, pixel_size, (0, sim_size[0]), (core_bottom, sim_size[1]), n_box)  # BOX below the core
add_waveguide(n, pixel_size, (0, sim_size[0]), (core_bottom + core_top) / 2, t_core, n_si)  # Si slab
# Etch the grating: paint the tooth GAPS back to air over the etched depth, on the
# air (top) side of the core.
gap_start = grating_start + duty * period  # first gap begins after the first tooth
add_grating(
    n, pixel_size,
    x_start=gap_start, period=period, duty=(1 - duty), n_teeth=n_teeth,
    y_span=(core_top, core_top + etch_depth), index=n_air,
)

permittivity = permittivity_2d(n)
nx, ny = permittivity.shape[:2]

# --- Source: true waveguide mode across the vertical cross-section ------------
i_src = int(round(x_source / pixel_size))
index_profile = np.sqrt(permittivity[i_src, :, 0])  # n(y) at the input cut
mode_profile, neff = solve_waveguide_mode(index_profile, pixel_size, wavelength, field="scalar")
print(f"Input waveguide fundamental mode: neff = {neff.real:.4f}")

# Estimated first-order emission angle from the grating equation (rough; uses the
# unetched-core neff, so treat as indicative).
sin_theta = (neff.real - wavelength / period) / n_air
emission_angle = np.degrees(np.arcsin(sin_theta)) if abs(sin_theta) <= 1 else float("nan")
print(f"Grating: period={period} um, {n_teeth} teeth -> estimated emission angle ~ {emission_angle:.1f} deg from normal")

source_values, source_position = mode_source(
    mode_profile,
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
    boundary_width=boundary_width,  # absorbing boundary width (μm)
    periodic=(False, False, True),  # 2D: uniform out of plane
    use_gpu=True,
)
print(f"Time {time() - start:.1f} s; Iterations {iterations}; Residual {residual_norm:.2e}")

# E_z field as a 2D (nx, ny) slice
ez = u[2]
if ez.ndim == 3:
    ez = ez[:, :, ez.shape[2] // 2]

# Restore a sane SIGINT handler before plotting (the cloud client's progress bar
# can leave it in a state that crashes matplotlib's window on close).
try:
    signal.signal(signal.SIGINT, signal.default_int_handler)
except (ValueError, TypeError):
    pass

# The money shot: guided light in the slab, then a beam diffracted up off the
# grating toward the fiber. We plot |E|^2 on a ~40 dB log scale with a fixed
# floor (a 1e-9 floor would just show numerical fog), and draw the core
# interfaces for context. output/ is git-ignored and auto-created.
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

intensity = np.abs(ez) ** 2
intensity /= intensity.max()
extent = [0, nx * pixel_size, ny * pixel_size, 0]

plt.figure(figsize=(11, 5))
plt.imshow(intensity.T, cmap="inferno", extent=extent, norm=LogNorm(vmin=1e-4, vmax=1.0), aspect="equal")
cbar = plt.colorbar(fraction=0.025, pad=0.02)
cbar.ax.set_title(r"$|E_z|^2$")
# Core interfaces and grating extent (dashed, faint) for context
for y_line in (core_top, core_bottom):
    plt.axhline(y_line, color="cyan", lw=0.5, alpha=0.4)
plt.xlabel(r"$x~(\mu m)$ -- propagation")
plt.ylabel(r"$y~(\mu m)$ -- air (top) / Si / BOX (bottom)")
plt.title(f"Grating coupler: guided mode diffracted to a free-space beam (~{emission_angle:.0f}deg)")
plt.tight_layout()

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.savefig(f"{OUTPUT_DIR}/maxwell_grating_coupler.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{OUTPUT_DIR}/maxwell_grating_coupler.pdf", bbox_inches="tight")
plt.show()
plt.close("all")
