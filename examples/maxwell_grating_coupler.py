"""
Maxwell 3D grating coupler (fiber-to-chip)
==========================================
A silicon-on-insulator (SOI) waveguide grating that lifts guided light off the
chip into a free-space beam aimed at a notional optical fiber. This is the
canonical photonic-integrated-circuit input/output structure: you cannot
butt-couple a ~10 um fiber mode to a sub-micron waveguide, so a periodic grating
diffracts the guided mode upward (out of plane) into a beam the fiber can catch.

This is a genuine 3D structure (no 2D side-view reduction): a finite-width 220 nm
silicon waveguide on a SiO2 buried oxide (BOX) with an air top-cladding, real
material indices, and a partial etch grating. The simulation axes are
  * x = propagation along the chip,
  * y = lateral (waveguide WIDTH direction),
  * z = vertical (the layer stack: BOX below, Si core, air above; the beam
        radiates up toward +z).

The fundamental waveguide mode is launched from the left in the unetched guide.
Where it reaches the partially-etched grating, each tooth scatters a little
power; the teeth add coherently into a single diffracted beam leaving the surface
at an angle set by the grating equation
``n_air sin(theta) = n_eff - lambda / period`` (first diffracted order).

Why a full-wave solver (not ray tracing): coupling is sub-wavelength tooth
scattering plus near-field overlap with the fiber mode -- diffraction, not
refraction. The Modified Born Series solver captures the whole near-field-to-
stand-off region in one CW solve. Quasi-TE polarization (dominant E along the
width, y) -- the usual silicon-photonics grating mode.

The launch field is the true waveguide mode from ``wsutils.svmodes`` on the real
(y, z) cross-section, not a Gaussian approximation.
"""

import os
import signal
from time import time

import numpy as np

from wavesimpro import simulate, create_monitor
from wsutils import svmodes, mode_source, OUTPUT_DIR

# --- Parameters (edit these to rescale) --------------------------------------
wavelength = 1.55  # design wavelength in micrometer (μm)
pixel_size = 0.02  # grid spacing (μm); ~6.5 px across the 130 nm etch, 30 px per period

n_si = 3.476   # silicon core
n_box = 1.444  # SiO2 buried oxide / cladding
n_air = 1.0    # air top cladding

t_core = 0.22   # Si core thickness, vertical z (μm)
W_wg = 2.5      # waveguide / grating width, lateral y (μm)

# Vertical layer stack (z increases upward; beam radiates toward +z into air).
box_thick = 1.30  # BOX thickness below the core (μm)
air_above = 3.60  # air above the core -- room for the beam to lift off (μm)
sim_z = box_thick + t_core + air_above
z_core_bot = box_thick
z_core_top = box_thick + t_core

# Lateral window
sim_y = W_wg + 2.0  # width + cladding/boundary margin (μm)
yc = sim_y / 2

# Grating (partial etch into the top of the core). Period chosen for an ~upward
# first-order beam; emission angle is reported from the solved mode neff.
period = 0.66    # grating pitch Λ (μm)
duty = 0.5       # tooth fraction of the period
etch_depth = 0.13  # partial etch depth from the core top (μm)
n_teeth = 18
grating_start = 3.0  # x where the grating begins (μm)

x_source = 1.6        # mode launch position (μm), clear of the boundary
boundary_width = 1.2  # absorbing boundary width (μm), all six faces

sim_x = grating_start + n_teeth * period + 3.0  # lead-in + grating + tail

nx = int(round(sim_x / pixel_size))
ny = int(round(sim_y / pixel_size))
nz = int(round(sim_z / pixel_size))
kz_box = int(round(z_core_bot / pixel_size))
kz_core_top = int(round(z_core_top / pixel_size))
kz_etch = int(round((z_core_top - etch_depth) / pixel_size))
jy0 = int(round((yc - W_wg / 2) / pixel_size))
jy1 = int(round((yc + W_wg / 2) / pixel_size))


def cross_section():
    """Permittivity n^2 of the unetched input (y, z) cross-section: W_wg x t_core Si
    core on a BOX, air above. Returns (nz, ny) = svmodes (z, y) convention."""
    n = np.full((nz, ny), n_air, dtype=np.complex64)
    n[:kz_box, :] = n_box                    # BOX below the core (all y)
    n[kz_box:kz_core_top, jy0:jy1] = n_si     # Si core (within the guide width)
    return (n ** 2)


# --- Launch mode + emission-angle estimate -----------------------------------
# Quasi-TE -> field 'EX' (E along horizontal/y = guide width).
neff, phi = svmodes(wavelength, n_si, 1, pixel_size, pixel_size, cross_section(), "0000", "EX")
e_in = phi.real  # (nz, ny)
print(f"Input waveguide fundamental mode: neff = {neff.real:.4f}")

sin_theta = (neff.real - wavelength / period) / n_air
emission_angle = np.degrees(np.arcsin(sin_theta)) if abs(sin_theta) <= 1 else float("nan")
print(f"Grating: period={period} um, {n_teeth} teeth -> estimated 1st-order angle ~ {emission_angle:.1f} deg from normal")

# --- Build the 3D geometry ---------------------------------------------------
permittivity_n = np.full((nx, ny, nz), n_air, dtype=np.complex64)
permittivity_n[:, :, :kz_box] = n_box                       # BOX slab (all x, y)
permittivity_n[:, jy0:jy1, kz_box:kz_core_top] = n_si        # Si waveguide (full x, guide width)
# Etch the grating: paint the tooth GAPS back to air over the etched depth, on the
# air (top) side of the core. First tooth at grating_start, then gaps after it.
for k in range(n_teeth):
    xa = grating_start + k * period + duty * period  # gap begins after the tooth
    xb = xa + (1 - duty) * period
    ia, ib = int(round(xa / pixel_size)), int(round(xb / pixel_size))
    permittivity_n[ia:ib, jy0:jy1, kz_etch:kz_core_top] = n_air
permittivity = permittivity_n ** 2
mvox = nx * ny * nz / 1e6
gb = mvox * 1e6 * 3 * 8 / 1e9  # vector field, complex64 (3 comp x 8 bytes)
print(f"Grid: {nx} x {ny} x {nz} = {mvox:.1f} Mvoxel  (px={pixel_size} um);  full-field download ~ {gb:.2f} GB")

# --- Source: launched waveguide mode over the (y, z) plane -------------------
source_values, source_position = mode_source(
    e_in.T.astype(np.complex64),       # (ny, nz)
    origin="topleft",
    position=[1, x_source, 0.0, 0.0],  # y-polarized (quasi-TE); profile fills the y,z window
    source_plane="yz",
    pixel_size=pixel_size,
)

y_mid = (jy0 + jy1) // 2

# --- Monitor (recommended): return just the x-z cut-plane this figure plots ---
# A monitor crops the result to this plane server-side, so only a thin slice is
# transferred instead of the full volume -- much faster, with identical values on
# the plotted plane. Keep it on; set USE_MONITOR = False only to fetch the whole
# 3D volume. The monitor is a thin slab through the guide centre; voxel index i on
# an n-voxel axis sits at (i + 0.5 - n/2)*px in domain-centred world coordinates.
USE_MONITOR = True
if USE_MONITOR:
    y_world = (y_mid + 0.5 - ny / 2) * pixel_size
    monitor_list = [create_monitor(
        name="cutplane",
        center_x=0.0, center_y=y_world, center_z=0.0,
        size_x=nx * pixel_size, size_y=3 * pixel_size, size_z=nz * pixel_size,
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
# vertical x-z slice through the guide centre (the mid y index).
ey = u[1]
field_side = ey[:, ey.shape[1] // 2, :]

try:
    signal.signal(signal.SIGINT, signal.default_int_handler)
except (ValueError, TypeError):
    pass

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

intensity = np.abs(field_side) ** 2
intensity /= intensity.max()
extent = [0, sim_x, 0, sim_z]

plt.figure(figsize=(11, 5))
plt.imshow(intensity.T, cmap="inferno", origin="lower", extent=extent, norm=LogNorm(vmin=1e-4, vmax=1.0), aspect="equal")
cbar = plt.colorbar(fraction=0.025, pad=0.02)
cbar.ax.set_title(r"$|E_y|^2$")
for z_line in (z_core_bot, z_core_top):
    plt.axhline(z_line, color="cyan", lw=0.5, alpha=0.4)
plt.xlabel(r"$x~(\mu m)$ -- propagation")
plt.ylabel(r"$z~(\mu m)$ -- BOX (bottom) / Si / air (top)")
plt.title(f"3D grating coupler: guided mode diffracted to a free-space beam (~{emission_angle:.0f} deg, guide-center x-z)")
plt.tight_layout()

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.savefig(f"{OUTPUT_DIR}/maxwell_grating_coupler.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{OUTPUT_DIR}/maxwell_grating_coupler.pdf", bbox_inches="tight")
plt.show()
plt.close("all")
