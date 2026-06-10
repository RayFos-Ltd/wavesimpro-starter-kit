import numpy as np
from typing import Sequence
from scipy.special import exp1
import matplotlib.pyplot as plt


def normalize(x: np.ndarray, min_val: float = None, max_val: float = None, a: float = 0, b: float = 1) -> float:
    """Normalize x to the range [a, b]

    :param x: Input array
    :param min_val: Minimum value (of x)
    :param max_val: Maximum value (of x)
    :param a: Lower bound for normalization
    :param b: Upper bound for normalization
    :return: Normalized x
    """
    if min_val is None:
        min_val = x.min()
    if max_val is None:
        max_val = x.max()
    normalized_x = (x - min_val) / (max_val - min_val) * (b - a) + a
    return normalized_x


def laplace_kernel_1d(pixel_size: float, length: int) -> np.ndarray:
    """Compute the kernel for the Laplace operator

    This function computes the exact Laplace kernel that corresponds to the second derivative of a sinc basis function,
    and then truncates it to a finite length. The truncation is done in such a way that the kernel still has
    an average value of zero, meaning that operating on a constant function will yield zero.

    Returns:
        The kernel for the Laplace operator in real space.
        The elements at negative positions are wrapped to the end of the array.
    """

    # original way (introduces wrapping artifacts in the kernel)
    # return -self.coordinates_f(dim) ** 2

    if length == 1:
        return np.zeros((1,))

    # x = [0, π, 2π .... -2π, -π]
    x = np.fft.fftfreq(length, 1 / (np.pi * length))

    # x_kernel = 2.0 * c / x**2 - 2.0 * s / x**3 + s / x
    # with all s=sin(x)=0
    x[0] = 1  # prevent division by 0
    x_kernel = 2.0 * np.cos(x) / x**2
    x_kernel[0] = 1.0 / 3.0  # remove singularity at x=0
    x_kernel *= -np.pi**2 / pixel_size**2

    # adjust end point(s) to ensure the kernel has an average value of zero
    # todo: find a more elegant way to do this
    if length % 2 == 0:
        x_kernel[length // 2] -= np.sum(x_kernel)  # even length, adjust only the furthest end point
    else:
        x_kernel[length // 2 : length // 2 + 1] -= 0.5 * np.sum(x_kernel)  # odd length, adjust the two end points

    return x_kernel


def diff_kernel_1d(pixel_size: float, length: int) -> np.ndarray:
    """Compute the kernel for the first derivative operator

    Returns:
        The kernel for the first derivative operator in real space.
        The elements at negative positions are wrapped to the end of the array.
    """
    x = np.fft.ifftshift(np.arange(-np.floor(length / 2), np.ceil(length / 2), dtype=np.float64)) * np.pi
    if x.size == 1:
        return x  # = [0.0]
    x[0] = 1  # prevent division by 0 warning
    x_kernel = np.pi / pixel_size * np.cos(x) / x
    x_kernel[0] = 0  # remove singularity at x=0
    return x_kernel


def _ulp(a: np.ndarray) -> float:
    """Returns the epsilon (machine precision error) for the data type of `a`"""
    try:
        finfo = np.finfo(a.dtype)
        exponent = np.ceil(np.log2(np.abs(a).max()))
        return float(finfo.eps) * 2.0 ** np.clip(exponent, finfo.minexp, finfo.maxexp)
    except ValueError:
        return 0.0


def plot_difference(a: np.ndarray, b: np.ndarray, tolerance: float):
    """Plot the difference between two arrays and the tolerance"""
    section = list(s // 2 for s in a.shape)
    if len(section) > 0:
        section[0] = slice(None)
    a = a[*section]
    b = b[*section]
    tolerance = tolerance[*section]

    plt.subplot(2, 1, 1)
    plt.plot(a.real, label="a.real")
    plt.plot(b.real, label="b.real")
    plt.plot(a.imag, label="a.imag")
    plt.plot(b.imag, label="b.imag")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(np.abs(a - b), label="|a-b|")
    plt.plot(np.arange(a.size), tolerance, label="tolerance")
    plt.legend()
    plt.show()


def all_close(a, b, rtol=0.0, atol=0.0, ulptol=100, plot=True):
    """Check if two tensors are close to each other.

    Condition: |a-b| <= atol + rtol * maximum(|b|,|a|) + ulptol * ulp
    Where ulp is the size of the smallest representable difference between two numbers of magnitude ~max(|b[...]|)
    Args:
        a: First tensor (NumPy array, or any array exposing a ``.gather()`` method, e.g. a wavesim engine array)
        b: Second tensor (NumPy array, or any array exposing a ``.gather()`` method)
        rtol: Relative tolerance
        atol: Absolute tolerance
        ulptol: Tolerance in units of the smallest representable difference between two numbers
        plot: If True, plot the difference between the two arrays if it exceeds the tolerances
    """
    # Duck-type engine arrays: if the object can ``gather()`` itself to host memory, do so;
    # otherwise treat it as array_like. This avoids a hard dependency on the wavesim engine.
    a = a.gather() if hasattr(a, "gather") else np.asarray(a)
    b = b.gather() if hasattr(b, "gather") else np.asarray(b)
    try:
        np.broadcast_shapes(a.shape, b.shape)
    except ValueError:
        print(f"Shapes do not match: {a.shape} != {b.shape}")
        return False
    if a.size == 0 and b.size == 0:
        return True

    # compute the size of a single ULP
    ab_max = np.maximum(np.abs(a), np.abs(b))  # max |a|, |b| for each element
    ulp = np.maximum(_ulp(a), _ulp(b))
    tolerance = atol + rtol * ab_max + ulptol * ulp  # tolerance per element
    diff = np.abs(a - b)
    if np.any(diff > tolerance):
        abs_err = diff.max().item()
        rel_err = (diff / ab_max).max()
        print(f"\nabsolute error {abs_err} = {abs_err / ulp} ulp\nrelative error {rel_err}")
        if plot:
            plot_difference(a, b, tolerance)
        return False
    else:
        return True


def max_abs_error(e, e_true):
    """(Normalized) Maximum Absolute Error (MAE) ||e-e_true||_{inf} / ||e_true||

    :param e: Computed field
    :param e_true: True field
    :return: (Normalized) MAE
    """
    return np.max(np.abs(e - e_true)) / np.linalg.norm(e_true)


def max_relative_error(e, e_true):
    """Computes the maximum error, normalized by the rms of the true field

    :param e: Computed field
    :param e_true: True field
    :return: (Normalized) Maximum Relative Error
    """
    return np.max(np.abs(e - e_true)) / np.sqrt(np.mean(np.abs(e_true) ** 2))


def relative_error(e, e_true):
    """Relative error ``⟨|e-e_true|^2⟩ / ⟨|e_true|^2⟩``

    :param e: Computed field
    :param e_true: True field
    :return: Relative Error
    """
    return np.nanmean(np.abs(e - e_true) ** 2) / np.nanmean(np.abs(e_true) ** 2)


def plot_fields(u, u_ref, re=None, normalize_u=True):
    """Plot the computed field u and the reference field u_ref.
    If u and u_ref are 1D arrays, the real and imaginary parts are plotted separately.
    If u and u_ref are 2D arrays, the absolute values are plotted.
    If u and u_ref are 3D arrays, the central slice is plotted.
    If normalize_u is True, the values are normalized to the same range.
    The relative error is (computed, if needed, and) displayed.
    """

    re = relative_error(u, u_ref) if re is None else re

    if u.ndim == 1 and u_ref.ndim == 1:
        plt.subplot(211)
        plt.plot(u_ref.real, label="Reference")
        plt.plot(u.real, label="Computed")
        plt.plot((u_ref.real - u.real)*10, label="Difference x 10")
        plt.legend()
        plt.title(f"Real part (RE = {relative_error(u.real, u_ref.real):.2e})")
        plt.grid()

        plt.subplot(212)
        plt.plot(u_ref.imag, label="Reference")
        plt.plot(u.imag, label="Computed")
        plt.plot((u_ref.imag - u.imag)*10, label="Difference x 10")
        plt.legend()
        plt.title(f"Imaginary part (RE = {relative_error(u.imag, u_ref.imag):.2e})")
        plt.grid()

        plt.suptitle(f"Relative error (RE) = {re:.2e}")
        plt.tight_layout()
        plt.show()
    else:
        if u.ndim == 4 and u_ref.ndim == 4:
            u = u[0, ...]
            u_ref = u_ref[0, ...]

        if u.ndim == 3 and u_ref.ndim == 3:
            u = u[u.shape[0] // 2, ...]
            u_ref = u_ref[u_ref.shape[0] // 2, ...]

        u = np.abs(u)
        u_ref = np.abs(u_ref)
        if normalize_u:
            min_val = min(np.min(u), np.min(u_ref))
            max_val = max(np.max(u), np.max(u_ref))
            a = 0
            b = 1
            u = normalize(u, min_val, max_val, a, b)
            u_ref = normalize(u_ref, min_val, max_val, a, b)
        else:
            a = None
            b = None

        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(u_ref, cmap="hot_r", vmin=a, vmax=b)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("Reference")

        plt.subplot(122)
        plt.imshow(u, cmap="hot_r", vmin=a, vmax=b)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("Computed")

        plt.suptitle(f"Relative error (RE) = {re:.2e}")
        plt.tight_layout()
        plt.show()


def relative_error_check(e, e_true, threshold=1.0e-3, plot=True):
    """Check if the relative error exceeds the threshold

    :param e: Computed field
    :param e_true: True field
    :param threshold: Threshold for the relative error
    :param plot: If True, plot the two arrays if the relative error exceeds the threshold
    :return: True if the relative error exceeds the threshold
    """
    re = relative_error(e, e_true)
    if re > threshold:
        print(f"Relative error {re:.2e} higher than threshold {threshold:.2e}")
        if plot:
            plot_fields(e, e_true, normalize_u=False)
        return False
    else:
        return True


def analytical_solution(x: Sequence[float], wavelength: float):
    """Compute analytic solution for 1D propagation in a homogeneous medium with refractive index 1

    The source is a sinc function centered at x=0

    Args:
        x: 1D array of positions to compute the solution at. Must be equally spaced and increasing
        wavelength: Wavelength
    """
    x = np.abs(np.asarray(x))
    h = np.abs(x[1] - x[0])  # pixel_size
    k = 2.0 * np.pi / wavelength
    phi = k * x

    u_theory = 1.0j * h / (2 * k) * np.exp(1.0j * phi) - h / (4 * np.pi * k) * (  # propagating plane wave
        np.exp(1.0j * phi) * (exp1(1.0j * (k - np.pi / h) * x) - exp1(1.0j * (k + np.pi / h) * x))
        - np.exp(-1.0j * phi) * (-exp1(-1.0j * (k - np.pi / h) * x) + exp1(-1.0j * (k + np.pi / h) * x))
    )
    small = np.abs(k * x) < 1.0e-10  # special case for values close to 0
    u_theory[small] = 1.0j * h / (2 * k) * (1 + 2j * np.arctanh(h * k / np.pi) / np.pi)  # exact value at 0.
    return u_theory
