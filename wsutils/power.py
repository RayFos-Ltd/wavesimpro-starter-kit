"""Power read-outs from a computed electric field: flux through a plane and absorbed power.

Both return power in the same units, |E|^2 * um^2 / Z0 with Z0 the impedance of free space,
so they can be summed into an energy budget: the power entering a closed box through its
faces equals the power leaving it plus the power absorbed inside. The fields follow the
exp(-i*omega*t) convention, in which exp(+i*k*x) travels towards +x.
"""

import numpy as np

# Central-difference weights for the first derivative, by number of points each side.
_STENCILS = {
    1: (1 / 2,),
    2: (2 / 3, -1 / 12),
    3: (3 / 4, -3 / 20, 1 / 60),
    4: (4 / 5, -1 / 5, 4 / 105, -1 / 280),
}


def _derivative(f: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    """First derivative along ``axis``: the widest central stencil that fits (up to 9 points),
    second order within its reach of the two ends."""
    n = f.shape[axis]
    out = np.gradient(f, spacing, axis=axis) if n > 1 else np.zeros_like(f)
    reach = min(4, (n - 1) // 2)
    if reach < 2:
        return out
    f = np.moveaxis(f, axis, 0)
    inner = sum(w * (f[reach + k : n - reach + k] - f[reach - k : n - reach - k])
                for k, w in enumerate(_STENCILS[reach], start=1)) / spacing
    np.moveaxis(out, axis, 0)[reach : n - reach] = inner
    return out


def power_flux(e: np.ndarray, pixel_size, wavelength: float, axis: int) -> float:
    """Time-averaged power through a plane normal to ``axis``, positive towards +axis.

    Args:
        e: Maxwell field ``(3, nx, ny, nz)`` on a slab, an odd number of voxels thick along
            ``axis``; the plane is its middle layer. H comes from the curl of E, so the slab
            needs at least 3 voxels, and more where the grid is coarse along ``axis``: 9
            voxels give the flux within 1% at 5 voxels per wavelength, 3 voxels need 25.
        pixel_size: voxel size in um, one value or one per axis.
        wavelength: vacuum wavelength in um.
        axis: 0, 1 or 2, the normal of the plane.
    """
    e = np.asarray(e)
    if e.ndim != 4 or e.shape[0] != 3:
        raise ValueError(f"e must have shape (3, nx, ny, nz), got {e.shape}")
    thickness = e.shape[1 + axis]
    if thickness < 3 or thickness % 2 == 0:
        raise ValueError(f"the slab must be an odd number (>= 3) of voxels thick along axis {axis}, got {thickness}")
    pitch = np.broadcast_to(np.asarray(pixel_size, dtype=float), (3,))
    k0 = 2 * np.pi / wavelength
    mid = thickness // 2
    u, v = (axis + 1) % 3, (axis + 2) % 3  # in-plane axes, (axis, u, v) right-handed

    def at_plane(a):
        return np.take(a, mid, axis=axis)

    def d(component, along):
        return at_plane(_derivative(e[component], pitch[along], along))

    # Z0 * H = curl(E) / (i * k0); only the in-plane components carry flux through the plane.
    h_u = (d(axis, v) - d(v, axis)) / (1j * k0)
    h_v = (d(u, axis) - d(axis, u)) / (1j * k0)
    s = 0.5 * np.real(at_plane(e[u]) * np.conj(h_v) - at_plane(e[v]) * np.conj(h_u))
    return float(s.sum() * pitch[u] * pitch[v])


def absorbed_power(e: np.ndarray, permittivity: np.ndarray, pixel_size, wavelength: float) -> float:
    """Time-averaged power absorbed in a volume: ``k0 / 2 * sum(Im(permittivity) * |E|^2) * dV``.

    Args:
        e: field over the volume, ``(3, nx, ny, nz)`` for Maxwell or ``(nx, ny, nz)`` for Helmholtz.
        permittivity: relative permittivity on the same voxels, ``(nx, ny, nz)``.
        pixel_size: voxel size in um, one value or one per axis.
        wavelength: vacuum wavelength in um.
    """
    e = np.asarray(e)
    intensity = np.abs(e) ** 2 if e.ndim == 3 else (np.abs(e) ** 2).sum(axis=0)
    voxel = float(np.prod(np.broadcast_to(np.asarray(pixel_size, dtype=float), (3,))))
    return float(np.pi / wavelength * (np.imag(permittivity) * intensity).sum() * voxel)
