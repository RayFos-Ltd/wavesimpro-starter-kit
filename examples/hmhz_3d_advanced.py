"""
Helmholtz 3D example to demonstrate simulate
============================================
This example demonstrates advanced functionalities of the simulate
helper function, by simulating the propagation of multiple sources
through a 3D medium with box-shaped block(s) of material(s) in a
homogeneous background permittivity, by solving the Helmholtz equation.
The example also defines and uses a simulation_callback.

Size: 0.704 Mvoxel including boundaries, 0.016 GB of memory (service estimate).
"""

import numpy as np
from time import time

from wsutils import create_medium, create_source, plot_computed, output_dir
from wavesimpro import simulate, configure

NAME = "hmhz_3d_advanced"  # run name on the service and output folder
OUTPUT_DIR = output_dir(NAME)
configure(description=NAME)

# Optical refractive index at 1 μm
material_PVP = np.complex64(1.5184 + 0.0011681j)  # PVP - Polyvinylpyrrolidone (a plastic)


def simulation_callback(domain, iteration, x, residual_norm, **kwargs):
    """Print the progress each time the service reports it; domain and x are not passed."""
    print(f"Iteration {iteration}: Residual Norm = {residual_norm:.2e}")


# Parameters
wavelength = 1.0  # Wavelength in micrometer (μm)
pixel_size = (
    wavelength / 4
)  # Pixel size in micrometer (μm). Divided by (at least) 3 the max of the absolute of the highest refractive index. wavelength / 4 is recommended.

# Size of the simulation domain
sim_size = np.array([12, 10, 15])  # Simulation size in micrometer (μm)

# Create a 3D permittivity map with a single block of a material, with all inputs in micrometer (μm)
material = material_PVP
permittivity = create_medium.cuboids_permittivity(
    sim_size, pixel_size=pixel_size, origin="center", origin_size_material=[([6, 2, 7.5], [6, 2, 7.5], material)]
)

# Sources term. A list of different sample sources at different locations is prepared. One or more can be combined in the simulation.
sourceList = []
point_source, ps_position = create_source.point_source(
    position=sim_size // 2, pixel_size=pixel_size  # source position in the center of the domain in micrometer (μm)
)  # Point source
gaussian_source, gs_position = create_source.gaussian_beam(
    shape=(sim_size[0], sim_size[2]),
    origin="topleft",  # source position is defined with respect to this origin
    position=[0, 0, 0],
    source_plane="xz",  # The string is always sorted by alphabetical order, so 'xz' and 'zx' are both recognized as 'xz'.
    pixel_size=pixel_size,
    amplitude=np.complex64(2.0, 0.0),
)  # Gaussian beam in XZ plane at Y start

# append the sources you want to use in the simulation
sourceList.append((point_source, ps_position))  # Point source in the center of the domain
sourceList.append((gaussian_source, gs_position))  # Gaussian beam in XZ plane at Y start
# sourceList.append( (gaussian_source, [0,permittivity.shape[1]//2,0]) ) # Gaussian beam in XZ plane at Y middle
# sourceList.append( (gaussian_source, [0,permittivity.shape[1],0]) ) # Gaussian beam in XZ plane at Y end

# Run the wavesim iteration and get the computed field
print("Running simulation...")
start = time()
u, iterations, residual_norm = simulate(
    permittivity=permittivity,
    sources=sourceList,
    wavelength=wavelength,
    pixel_size=pixel_size,
    periodic=(False, False, False),
    boundary_width=5,  # Boundary width in micrometer (μm)
    callback=simulation_callback,
)
sim_time = time() - start
print(f"Time {sim_time:2.2f} s; Iterations {iterations}; Time per iteration {sim_time / iterations:.4f} s")
print(f"(Residual norm {residual_norm:.2e})")

plot_computed(u, pixel_size, save=True, filename=f"{OUTPUT_DIR}/{NAME}")
