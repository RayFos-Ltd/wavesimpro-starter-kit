"""Carry a field from one plane to a parallel plane through a homogeneous medium."""

import numpy as np


def propagate(field: np.ndarray, pixel_size, wavelength: float, distance: float,
              refractive_index: float = 1.0, padding: int = 1) -> np.ndarray:
    """The field on a plane ``distance`` further along the normal, by its angular spectrum.

    The plane is split into plane waves, each is advanced by its own phase and they are
    summed again. This is exact in a homogeneous medium, so empty space behind a structure
    does not have to be part of the simulation. Evanescent waves are dropped.

    Args:
        field: complex field with the plane on its last two axes, ``(..., n1, n2)``; leading
            axes (for example the three components of a Maxwell field) are carried along.
        pixel_size: sample spacing on the plane in um, one value or one per plane axis.
        wavelength: vacuum wavelength in um.
        distance: how far to carry the field, in um; negative carries it backwards.
        refractive_index: real index of the medium.
        padding: zeros added on each plane axis, in units of the plane's own size, so light
            leaving one side does not come back in on the other. Use more when the beam
            spreads beyond the plane.

    Returns:
        The field on the new plane, same shape as ``field``.
    """
    field = np.asarray(field)
    n1, n2 = field.shape[-2:]
    d1, d2 = np.broadcast_to(np.asarray(pixel_size, dtype=float), (2,))
    m1, m2 = n1 * (1 + padding), n2 * (1 + padding)
    k = 2 * np.pi * refractive_index / wavelength
    k1 = 2 * np.pi * np.fft.fftfreq(m1, d1)[:, None]
    k2 = 2 * np.pi * np.fft.fftfreq(m2, d2)[None, :]
    kn2 = k ** 2 - k1 ** 2 - k2 ** 2  # squared wavenumber along the normal
    advance = np.where(kn2 > 0, np.exp(1j * np.sqrt(np.maximum(kn2, 0)) * distance), 0)
    spectrum = np.fft.fft2(field, s=(m1, m2), axes=(-2, -1))
    return np.fft.ifft2(spectrum * advance, axes=(-2, -1))[..., :n1, :n2]
