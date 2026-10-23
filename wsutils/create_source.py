import numpy as np
from typing import Sequence
from scipy.signal.windows import gaussian

# Local lightweight alias (formerly wavesim.engine.index_utils.shape_like).
# These helpers build and return plain NumPy arrays, so no engine type is needed.
shape_like = Sequence[int]


def point_source(
    position: Sequence[float],
    pixel_size: float,
    amplitude: complex = 1.0,
    ):
    """Define a point source with given amplitude (default 1), and position
    Args:
        position: a tuple of 3 or 4 floats, the position of the center of the source in micrometer (μm).
            3 floats (x, y, z) for solving the Helmholtz equation (scalar).
            4 floats (polarization axis, x, y, z) for solving time-harmonic Maxwell's equations for non-magnetic and non-birefringent materials (vector). Polarization axis must be 0, 1, or 2 for x, y, or z polarization, respectively.
        pixel_size: pixel size in micrometer (μm)
        amplitude: amplitude to scale the source by. Default is 1.0
    Returns:
        source_values: 3D (4D) array of complex numbers with shape (1, 1, 1) ((1, 1, 1, 1))
        position: a tuple of 3 (4) integers indicating (the polarization and) the position of the source in pixels.
    """
    check_source_parameters(position)
    p = polarization_(position)
    shape = (1, 1, 1, 1) if p else (1, 1, 1)
    position = np.asarray(position)
    position[p:] = position[p:] / pixel_size
    position = tuple(position.astype(int))
    return np.full(shape=shape, fill_value=amplitude, dtype=np.complex64), position


def plane_wave(
    shape: shape_like,
    origin: str,
    position: Sequence[float],
    source_plane: str,
    pixel_size: float,
    amplitude: complex = 1.0,
    theta: float = None,
    phi: float = None,
    wavelength: float = None,
):
    """Define a plane wave source. The source can be given an incident angle theta
    or azimuthal angle phi in radians.

    Args:
        shape: shape of the source (3D array) in micrometer (μm)
        origin (str): Define the origin as 'center' or 'topleft.' The position of the source is defined with respect to the origin.
        position: a tuple of 3 or 4 floats, the position of the center/top left of the source in micrometer (μm).
            3 floats (x, y, z) for solving the Helmholtz equation (scalar).
            4 floats (polarization axis, x, y, z) for solving time-harmonic Maxwell's equations for non-magnetic and non-birefringent materials (vector). Polarization axis must be 0, 1, or 2 for x, y, or z polarization, respectively.
        source_plane: plane to define the source in. should be a string of one or two axes from x, y, and z.
            For example, for a 2D plane wave source in the xz plane, source_plane='xz'.
            The string is always sorted by alphabetical order, so 'xy' and 'yx' are both recognized as 'xy'.
        pixel_size: pixel size in micrometer (μm)
        amplitude: amplitude to scale the source by. Default is 1.0
        theta: angle of incidence of the source with respect to the source axis in radians
        phi: angle of incidence of the source in the plane orthogonal to the source axis in radians
        wavelength: wavelength in micrometer (μm). Must be defined when theta or phi is not None.
    Returns:
        source_values: 3D array of complex numbers with shape determined by shape and source_plane
        position: a tuple of 3 (4) integers indicating (the polarization and) the position of the source in pixels.
    """
    if origin not in ['center', 'topleft']:
        raise ValueError(f"Specify origin from the options ['center', 'topleft']. Invalid origin: '{origin}'")
    source_plane = ''.join(sorted(source_plane.lower()))

    check_source_parameters(position, shape, source_plane)
    p = polarization_(position)

    # convert shape from micrometers to pixels
    shape = (np.asarray(shape) / pixel_size).astype(int)
    shape = (shape,) if isinstance(shape, int) or isinstance(shape, np.int64) else tuple(shape)

    source_values = np.ones(shape=shape[0], dtype=np.complex64)
    if theta is not None:
        assert wavelength is not None, "wavelength must be given for an angled source with some theta"
        source_values = source_angled(source_values, theta, wavelength, pixel_size)

    if len(shape) == 2:  # a 2D plane wave
        source_values_ = np.ones(shape=shape[1], dtype=np.complex64)
        if phi is not None:
            assert wavelength is not None, "wavelength must be given for an angled source with some phi"
            source_values_ = source_angled(source_values_, phi, wavelength, pixel_size)
        source_values = source_values[:, None] * source_values_  # 2D plane wave

    source_values *= amplitude
    assert source_values.shape == shape, f"Source shape {source_values.shape} does not match expected shape {shape}"
    source_values = expand_source_shape(source_values, source_plane)

    source_values = source_values[None, ...] if p else source_values  # Add polarization dimension if polarization (p) is True
    if p:
        assert source_values.ndim == 4, f"Expected source shape {source_values.shape} to be 4D"
    else:
        assert source_values.ndim == 3, f"Expected source shape {source_values.shape} to be 3D"
    position = np.asarray(position)
    position[p:] = position[p:] / pixel_size
    if origin == 'center':
        for i, axis in enumerate(source_plane):
            position[p + "xyz".index(axis)] -= shape[i] // 2
    position = tuple(position.astype(int))

    return source_values.astype(np.complex64), position


def gaussian_beam(
    shape: shape_like,
    origin: str,
    position: Sequence[float],
    source_plane: str,
    pixel_size: float,
    amplitude: complex = 1.0,
    theta: float = None,
    phi: float = None,
    wavelength: float = None,
    waist: float = None,
):
    r"""Define a plane wave source with a Gaussian intensity profile: the field amplitude
    :math:`w(n) = e^{-n^2 / 2\sigma^2}` sampled over ``shape``.

    The Gaussian beam can be given an incident angle theta or azimuthal angle phi in radians.

    Args:
        shape: shape of the source (3D array) in micrometer (μm)
        origin (str): Define the origin as 'center' or 'topleft.' The position of the source is defined with respect to the origin.
        position: a tuple of 3 or 4 floats, the position of the center/top left of the source in micrometer (μm).
            3 floats (x, y, z) for solving the Helmholtz equation (scalar).
            4 floats (polarization axis, x, y, z) for solving time-harmonic Maxwell's equations for non-magnetic and non-birefringent materials (vector). Polarization axis must be 0, 1, or 2 for x, y, or z polarization, respectively.
        source_plane: plane to define the source in. should be a string of one or two axes from x, y, and z.
            For example, for a 2D Gaussian beam source in the xz plane, source_plane='xz'.
            The string is always sorted by alphabetical order, so 'xy' and 'yx' are both recognized as 'xy'.
        pixel_size: pixel size in micrometer (μm)
        amplitude: amplitude to scale the source by. Default is 1.0
        theta: angle of incidence of the source with respect to the source axis in radians
        phi: angle of incidence of the source in the plane orthogonal to the source axis in radians
        wavelength: wavelength in micrometer (μm). Must be defined when theta or phi is not None.
        waist: beam waist w0 in micrometer (μm) -- the 1/e^2 intensity radius, so
            sigma = w0 / sqrt(2), the field FWHM is 1.665 w0 and the intensity FWHM
            1.177 w0. A 2D beam uses the same waist on both axes. Default: the widest
            beam that still falls to 1.1 % of its peak at the edge of ``shape``; a wider
            one leaves a step there for ``apodise`` to taper.
    Returns:
        source_values: 3D array of complex numbers with shape determined by shape and source_plane
        position: a tuple of 3 (4) integers indicating (the polarization and) the position of the source in pixels.
    """
    if origin not in ['center', 'topleft']:
        raise ValueError(f"Specify origin from the options ['center', 'topleft']. Invalid origin: '{origin}'")
    source_plane = ''.join(sorted(source_plane.lower()))

    check_source_parameters(position, shape, source_plane)
    p = polarization_(position)

    # convert shape from micrometers to pixels
    shape = (np.asarray(shape) / pixel_size).astype(int)
    shape = (shape,) if isinstance(shape, int) or isinstance(shape, np.int64) else tuple(shape)

    m = shape[0]  # number of points in the Gaussian window
    if waist is None:
        # (m-1)/2 is centre to edge; 3 puts that edge at 3 sigma. 3 was the default of the
        # retired alpha/n_sigma parameter, and the value every call site passed.
        sigma = (m - 1) / (2 * 3)
    else:
        sigma = waist / (np.sqrt(2) * pixel_size)  # w0 = sqrt(2) sigma, in pixels
    source_values = gaussian(m, sigma).astype(np.complex64)
    if theta is not None:
        assert wavelength is not None, "wavelength must be given for an angled source with some theta"
        source_values = source_angled(source_values, theta, wavelength, pixel_size)

    if len(shape) == 2:  # a 2D Gaussian beam
        source_values_ = gaussian(shape[1], sigma).astype(np.complex64)
        if phi is not None:
            assert wavelength is not None, "wavelength must be given for an angled source with some phi"
            source_values_ = source_angled(source_values_, phi, wavelength, pixel_size)
        source_values = source_values[:, None] * source_values_  # 2D Gaussian beam

    source_values *= amplitude
    assert source_values.shape == shape, f"Source shape {source_values.shape} does not match expected shape {shape}"
    source_values = expand_source_shape(source_values, source_plane)

    source_values = source_values[None, ...] if p else source_values  # Add polarization dimension if polarization (p) is True
    if p:
        assert source_values.ndim == 4, f"Expected source shape {source_values.shape} to be 4D"
    else:
        assert source_values.ndim == 3, f"Expected source shape {source_values.shape} to be 3D"
    position = np.asarray(position)
    position[p:] = position[p:] / pixel_size
    if origin == 'center':
        for i, axis in enumerate(source_plane):
            position[p + "xyz".index(axis)] -= shape[i] // 2
    position = tuple(position.astype(int))

    return source_values.astype(np.complex64), position


def source_angled(source_values, angle, wavelength, pixel_size):
    kx = 2 * np.pi / wavelength * np.sin(angle)
    x = np.arange(1, source_values.shape[0] + 1) * pixel_size
    return source_values * np.exp(1j * kx * x)


def expand_source_shape(source_values, plane: str):
    if plane == 'x':
        return np.expand_dims(source_values, axis=(1, 2))
    elif plane == 'y':
        return np.expand_dims(source_values, axis=(0, 2))
    elif plane == 'z':
        return np.expand_dims(source_values, axis=(0, 1))
    elif plane == 'xy':
        return np.expand_dims(source_values, axis=(2))
    elif plane == 'yz':
        return np.expand_dims(source_values, axis=(0))
    elif plane == 'xz':
        return np.expand_dims(source_values, axis=(1))
    else:
        raise ValueError("source_plane should be in ['x', 'y', 'z', 'xy', 'yz', 'xz']")


def check_source_parameters(position, shape=None, source_plane=None):
    if len(tuple(position)) != 3 and len(tuple(position)) != 4:
        raise ValueError(f"Source position {position} invalid, should be a tuple of 3 (or 4) floats indicating the (polarization axis and the) position in x, y, and z axes in micrometer (μm)")

    if shape is not None and source_plane is not None:
        if source_plane in ['x', 'y', 'z']:
            if not (isinstance(shape, (int, np.integer)) or len(tuple(shape)) == 1):
                raise ValueError(f"Source shape {shape} invalid, should be 1D for source_plane='{source_plane}'")
        elif source_plane in ['xy', 'yz', 'xz']:
            assert len(shape) == 2, f"Source shape {shape} invalid, should be 2D for source_plane='{source_plane}'"
        else:
            raise ValueError("source_plane should be in ['x', 'y', 'z', 'xy', 'yz', 'xz']")


def polarization_(position):
    """Check whether there is a polarization axis (1) or not (0) and return it"""
    p = int(len(tuple(position)) == 4)  # polarization True (1) or False (0)
    if p:
        assert position[0] in [0, 1, 2], f"polarization axis, i.e., position[0], must be 0, 1, or 2 for x, y, or z polarization, respectively"
    return p


# --- Source modifiers --------------------------------------------------------
# Two different jobs, and it is easy to assume one buys the other:
#
#   apodise()          smooths a source envelope along any axis, suppressing the spectral
#                      sidelobes a hard edge creates. Says nothing about direction.
#   make_directional() drives a LENGTH of guide with the mode's own exp(i*beta*s), tapered
#                      at its ends, so it is phase-matched to the wave travelling one way
#                      and mismatched to the one travelling the other.
#
# Measured on a straight 500 x 220 nm SOI guide at 1550 nm, 15 cells/wavelength, reading
# the guided power behind the source as a fraction of the forward power:
#
#     plane sheet, no taper                  101.2 % backward
#     plane sheet, tapered                   101.0 % backward
#     1.5 um volume, phase-matched, alpha=0    0.38 % backward
#     3.0 um volume, phase-matched, alpha=0    0.11 % backward
#     3.0 um volume, phase-matched, alpha=1    0.00 % backward
#
# Tapering a plane sheet changes the backward power by 0.2 points, i.e. not at all: a
# plane electric current sheet has no magnetic partner and radiates both ways by symmetry.
# Only the phase ramp fixes that. The taper meanwhile improved the near-field flatness in
# both cases (3.2 % -> 2.6 % for the sheet, 0.80 % -> 0.14 % for the volume), which is the
# job it does do.


def _window_1d(n, alpha, kind):
    """One-dimensional taper of length n. Tukey is the default family because it spans
    the useful range: alpha 0 is rectangular, alpha 1 is Hann."""
    if n < 3:
        return np.ones(max(n, 1))  # fewer than three samples have no interior to taper
    if kind == "tukey":
        from scipy.signal.windows import tukey as _tukey
        return _tukey(n, min(max(alpha, 0.0), 1.0))
    if kind == "hann":
        return np.hanning(n + 2)[1:-1]
    raise ValueError(f"window must be 'tukey' or 'hann', got {kind!r}")


def apodise(source, axes, alpha=0.3, window="tukey"):
    """Taper a source's envelope along one or more axes.

    A source that stops abruptly has a spectrum with slowly decaying tails, so it radiates
    angular content nobody asked for -- the hard-aperture diffraction that several examples
    in this kit already avoid by multiplying a Tukey window onto their plane waves. This
    does that in one place, for any source, including ones built by hand.

    It does NOT make a source directional; see ``make_directional``.

    Args:
        source: ``(source_values, position)`` as every source helper here returns.
        axes: the axes to taper: 'x', 'y' or 'z', or several as one string ('yz') or a
            sequence. Use the transverse axes for a sheet or beam, the propagation axis
            for the ends of a volume source built by ``make_directional``.
        alpha: Tukey taper fraction, 0 (no taper) to 1 (full Hann). 0.2-0.35 is what the
            plane-wave examples here use.
        window: 'tukey' (default) or 'hann'.
    Returns:
        ``(source_values, position)`` with the same shapes and position.
    """
    values, position = source
    values = np.asarray(values).copy()
    p = polarization_(position)
    for ax in tuple(axes):
        if ax not in "xyz" or len(ax) != 1:
            raise ValueError(f"axes entries must be 'x', 'y' or 'z', got {ax!r}")
        k = p + "xyz".index(ax)
        n = values.shape[k]
        if n < 2:
            continue        # a singleton axis has no envelope to taper
        w = _window_1d(n, alpha, window)
        if values.dtype.kind in "fc":
            w = w.astype(values.real.dtype)  # taper in the source's own precision
        values *= w.reshape([-1 if i == k else 1 for i in range(values.ndim)])
    return values.astype(np.complex64), position


def make_directional(source, axis, n_eff, wavelength, pixel_size, length,
                     direction=+1, alpha=1.0):
    """Turn a planar source into a phase-matched volume source that launches one way.

    A single electric current sheet radiates equally both ways -- it is missing the
    magnetic surface current that, in the equivalence principle, cancels one side. Rather
    than supply that (the solver takes an electric source only), this drives a LENGTH of
    the guide with the mode's own ``exp(i*beta*s)``. The forward wave is phase-matched
    along the whole source and adds; the backward wave is mismatched by ``2*beta`` and
    cancels. Radiation, travelling at a different phase velocity, is suppressed the same
    way.

    Measured on a 500 x 220 nm SOI guide at 1550 nm with the default taper, 3.0 um of
    source leaves no measurable power going backwards; untapered (``alpha=0``), 1.5 um
    leaves 0.38 % of the forward power and 3.0 um leaves 0.11 %. Longer is more
    directional, at the cost of occupying more of the domain.

    Placement, also measured on that guide:
      * how much clearance a device or monitor needs past the source END depends on how
        faithfully the source was built, not on this function. Driven with the current
        that launches the mode -- ``J = n_hat x H`` from a full-vector mode solver --
        the power read downstream is within 0.2 % of its settled value a quarter of a
        micron out. Driven with an approximation to that current it can be a percent or
        two out for several microns, and anything placed inside that region is being lit
        by light that is not the mode. Measure it rather than assume it: read the guided
        power at a series of planes on plain guide and see where it stops changing.
      * the absorbing boundary does not perturb the launch even with the source's leading
        edge 0.15 um from the domain face (forward power flat to 0.06 % over a standoff
        sweep), because the absorber is added OUTSIDE the domain and the source is still
        in lossless material.

    Args:
        source: ``(source_values, position)``; must be a single plane along ``axis``.
        axis: 'x', 'y' or 'z' -- the propagation axis.
        n_eff: effective index of the mode being launched. Required: the phase matching IS
            the mechanism, and a ramp built on the wrong beta is worse than no ramp.
        wavelength: vacuum wavelength, same length unit as ``pixel_size``.
        pixel_size: grid spacing.
        length: how much guide to drive, same unit. About 2-3 guide wavelengths
            (``wavelength / n_eff``) is enough; below that directionality falls off.
        direction: +1 to launch towards increasing ``axis``, -1 for decreasing.
        alpha: Tukey fraction of the taper on the driven length's ends: 1.0 is a full Hann
            window, 0 is uniform. The ends of a uniform driven region radiate.
    Returns:
        ``(source_values, position)`` -- values gain ``round(length/pixel_size)`` planes
        along ``axis`` and position moves back by half of them (rounded down), so the
        plane the source was given on keeps phase zero; for an even count the extra plane
        lies upstream. The plane's current is spread over the ramp with weights that sum
        to 1, so the forward launch amplitude depends on neither ``length`` nor
        ``pixel_size``.
    """
    values, position = source
    values = np.asarray(values)
    p = polarization_(position)
    if axis not in ("x", "y", "z"):
        raise ValueError(f"axis must be 'x', 'y' or 'z', got {axis!r}")
    k = p + "xyz".index(axis)
    if values.shape[k] != 1:
        raise ValueError(f"source is already {values.shape[k]} samples along {axis}; "
                         f"make_directional expects a single plane")
    n = max(int(round(length / pixel_size)), 1)
    s = (np.arange(n) - n // 2) * pixel_size  # s = 0 on the plane the source came on
    w = _window_1d(n, alpha, "tukey")
    ramp = w / w.sum() * np.exp(1j * direction * 2 * np.pi * n_eff / wavelength * s)
    values = values * ramp.reshape([-1 if i == k else 1 for i in range(values.ndim)])

    position = list(position)
    position[k] = int(position[k]) - n // 2
    return values.astype(np.complex64), tuple(position)
