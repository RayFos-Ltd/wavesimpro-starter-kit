"""
A first waveguide: light stays in a dielectric slab
===================================================
A 2-D silicon-nitride strip (n = 2.0, 0.5 μm wide) in silica (n = 1.444) runs along x.
A narrow Gaussian beam launched into the strip near its start is guided by total internal
reflection, so the field at the far end is still confined to the strip. The script prints
how much of the power stays within the strip at two positions and plots |E_z|.

Couplers, splitters and devices read from GDSII layouts are in the companion repository
``wavesimpro-examples``.
"""

import numpy as np
from time import time

from wavesimpro import simulate
from wsutils.create_source import gaussian_beam
from wsutils import plot_computed

# Parameters
wavelength = 1.55  # micrometer (μm)
n_core, n_clad = 2.0, 1.444
core_width = 0.5  # μm
pixel_size = 0.05  # μm: 10 voxels across the strip, about 15 per wavelength in the core
sim_size = np.array([20.0, 6.0])  # (x, y) in μm

# Permittivity: cladding everywhere, the strip along x through the middle
nx, ny = (sim_size / pixel_size).round().astype(int)
y = (np.arange(ny) + 0.5) * pixel_size - sim_size[1] / 2
permittivity = np.full((nx, ny), n_clad ** 2, dtype=np.complex64)
permittivity[:, np.abs(y) < core_width / 2] = n_core ** 2

# A beam across the strip at x = 2 μm, polarized along z (out of the plane)
source_values, source_position = gaussian_beam(
    shape=(1.5,),  # source width across the strip, in μm
    origin="center",
    position=[2, 2.0, sim_size[1] / 2, 0],  # (polarization, x, y, z) in μm
    source_plane="y",
    pixel_size=pixel_size,
    waist=0.3,  # μm, close to the strip width
)

print("Running simulation...")
start = time()
u, iterations, residual_norm = simulate(
    permittivity=permittivity[..., None],
    sources=[(source_values, source_position)],
    wavelength=wavelength,
    pixel_size=pixel_size,
    boundary_width=4 * wavelength,
    periodic=(False, False, True),
)
print(f"Time {time() - start:2.2f} s; Iterations {iterations}; Residual norm {residual_norm:.2e}")

# Share of |E|^2 inside the strip at two positions along it
intensity = (np.abs(u) ** 2).sum(axis=0)  # (x, y)
in_core = np.abs(y) < core_width / 2
for x_um in (4.0, 18.0):
    ix = int(x_um / pixel_size)
    share = intensity[ix, in_core].sum() / intensity[ix].sum()
    print(f"x = {x_um:4.1f} um: {share * 100:5.1f}% of the power is inside the strip")

plot_computed(u[2], pixel_size, normalize_u=False)
