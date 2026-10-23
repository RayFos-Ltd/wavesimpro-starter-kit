"""
Monitors: ask for parts of the field instead of the whole volume
================================================================
A Gaussian beam hits a dielectric sphere. Instead of downloading the full 3-D field,
two monitors are requested: an x-z plane through the sphere centre and a small cube
behind it. With ``monitors=`` set, ``simulate`` returns a list with one array per
monitor, in the order given, each shaped (3, nx, ny, nz) like the full field.
"""

import numpy as np
from time import time

from wavesimpro import simulate, create_monitor
from wsutils.create_medium import sphere_permittivity
from wsutils.create_source import gaussian_beam
from wsutils import plot_computed

# Parameters
wavelength = 1.0  # micrometer (μm)
pixel_size = wavelength / 4
sim_size = 12.0  # cube edge in micrometer (μm)
n_voxels = int(round(sim_size / pixel_size))

# A glass sphere (n = 1.5) of radius 2 μm in air, centred in the domain; permittivity = n^2
permittivity, *_ = sphere_permittivity((n_voxels, n_voxels, n_voxels), pixel_size, 2.0, 1.5 ** 2, 1.0)

# A beam travelling along +x, polarized along y, entering 1 μm inside the domain
source_values, source_position = gaussian_beam(
    shape=(sim_size, sim_size),
    origin="center",
    position=[1, 1.0, sim_size / 2, sim_size / 2],  # (polarization, x, y, z) in μm
    source_plane="yz",
    pixel_size=pixel_size,
    waist=2.0,
)

# Monitor positions are in world coordinates with the origin at the domain centre;
# a size of one pixel along an axis returns a single voxel on that axis.
plane = create_monitor(name="xz_plane", center_x=0.0, center_y=0.0, center_z=0.0,
                       size_x=sim_size, size_y=pixel_size, size_z=sim_size)
cube = create_monitor(name="behind_sphere", center_x=3.5, center_y=0.0, center_z=0.0,
                      size_x=2.0, size_y=2.0, size_z=2.0)

print("Running simulation...")
start = time()
fields, iterations, residual_norm = simulate(
    permittivity=permittivity,
    sources=[(source_values, source_position)],
    wavelength=wavelength,
    pixel_size=pixel_size,
    boundary_width=4 * wavelength,
    monitors=[plane, cube],
)
print(f"Time {time() - start:2.2f} s; Iterations {iterations}; Residual norm {residual_norm:.2e}")

u_plane, u_cube = fields
print(f"xz_plane monitor: {u_plane.shape} (polarization, x, y, z)")
print(f"behind_sphere monitor: {u_cube.shape}")
print(f"mean |E|^2 behind the sphere: {np.mean(np.abs(u_cube) ** 2):.3e}")

# |E|^2 on the plane, summed over the three components
plot_computed(np.sqrt((np.abs(u_plane) ** 2).sum(axis=0))[:, 0, :], pixel_size, normalize_u=False, axis_labels=("x", "z"))
