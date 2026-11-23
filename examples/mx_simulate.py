"""
Maxwell simulate
====================
This example demonstrates all the arguments of the simulate wrapper
function through an example solving Maxwell's equations.

Size: 2.765 Mvoxel including boundaries, 0.144 GB of memory (service estimate).
"""

import numpy as np
from time import time
import matplotlib.pyplot as plt

from wsutils.create_medium import random_permittivity
from wsutils.create_source import gaussian_beam
from wavesimpro import simulate, configure
from wsutils import plot_computed, output_dir

NAME = "mx_simulate"  # run name on the service and output folder
OUTPUT_DIR = output_dir(NAME)
configure(description=NAME)


def simulation_callback(domain, iteration, x, residual_norm, **kwargs):
    """Print the progress each time the service reports it; domain and x are not passed."""
    print(f"Iteration {iteration}: Residual Norm = {residual_norm:.2e}")


# Parameters
wavelength = 1  # Wavelength in micrometer (μm)
pixel_size = wavelength / 8  # Pixel size in micrometer (μm)

# Size of the simulation domain
sim_size = np.array([8, 5, 10])  # Simulation size in micrometer (μm)

# Create permittivity map
permittivity = random_permittivity(sim_size, pixel_size, seed=42)

# Create a plane wave source with Gaussian intensity profile
source_values, source_position = gaussian_beam(
    shape=(sim_size[1], sim_size[2]),  # source shape in micrometer (μm)
    origin="topleft",  # source position is defined with respect to this origin   
    position=[1, 0, 0, 0],  # (polarization, x, y, z): polarization axis 0/1/2 = x/y/z; x, y, z = top-left corner of the source in micrometer (μm)
    source_plane="yz",  # The string is always sorted by alphabetical order, so 'yz' and 'zy' are both recognized as 'yz'.
    pixel_size=pixel_size,
)

# Run the wavesim iteration and get the computed field
print("Running simulation...")
start = time()
u, iterations, residual_norm = simulate(
    permittivity=permittivity,
    sources=[(source_values, source_position)],
    wavelength=wavelength,
    pixel_size=pixel_size,
    boundary_width=5,  # Boundary width in micrometer (μm)
    periodic=(False, False, False),
    max_iterations=10000,
    threshold=1.0e-7,  # tolerance for convergence. Default is 1.0e-6.
    alpha=0.9,  # relaxation factor for the preconditioned Richardson method. Default is 0.75.
    full_residuals=True,  # if true return full residuals, and if False, return only the final residual norm. Default is False.
    callback=simulation_callback,
)
sim_time = time() - start
print(f"Time {sim_time:2.2f} s; Iterations {iterations}; Time per iteration {sim_time / iterations:.4f} s")
print(f"(Residual norm {residual_norm[-1]:.2e})")

# Plot the results
plot_computed(u[1, ...], pixel_size, log=True, save=True, filename=f"{OUTPUT_DIR}/{NAME}")

# Plot the residual norm with iterations
plt.semilogy(np.arange(1, len(residual_norm) + 1), residual_norm)
plt.xlabel("Iterations")
plt.ylabel("Residual norm")
plt.savefig(f"{OUTPUT_DIR}/{NAME}_residual.png", dpi=300, bbox_inches="tight")
plt.show()
