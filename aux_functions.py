from scipy.fftpack import fft, fft2, fftn, fftshift, ifft, ifft2, ifftn, ifftshift
import numpy as np

# auxilliary functions from ASPIRE

# from utils/fft.py

def centered_ifft2(x):
    """
    Calculate a centered, two-dimensional inverse FFT
    :param x: The two-dimensional signal to be transformed.
        The inverse FFT is only applied along the last two dimensions.
    :return: The centered inverse Fourier transform of x.
    """

    x = ifftshift(x, axes=(-2, -1))

    x = ifft2(x, axes=(-2, -1))

    x = fftshift(x, axes=(-2, -1))

    return x

def centered_fft2(x):
    """
    Calculate a centered, two-dimensional inverse FFT
    :param x: The two-dimensional signal to be transformed.
        The inverse FFT is only applied along the last two dimensions.
    :return: The centered inverse Fourier transform of x.
    """

    x = ifftshift(x, axes=(-2, -1))

    x = fft2(x, axes=(-2, -1))

    x = fftshift(x, axes=(-2, -1))

    return x


# from utils/coord_trans.py

def cart2pol(x, y):
    """
    Convert Cartesian to Polar Coordinates. All input arguments must be the same shape.
    :param x: x-coordinate in Cartesian space
    :param y: y-coordinate in Cartesian space
    :return: A 2-tuple of values:
        theta: angular coordinate/azimuth
        r: radial distance from origin
    """
    return np.arctan2(y, x), np.hypot(x, y)

def grid_2d(n, shifted=False, normalized=True, dtype=np.float32):
    """
    Generate two dimensional grid.
    :param n: the number of grid points in each dimension.
    :param shifted: shifted by half of grid or not when n is even.
    :param normalized: normalize the grid in the range of (-1, 1) or not.
    :return: the rectangular and polar coordinates of all grid points.
    """
    grid = np.ceil(np.arange(-n / 2, n / 2, dtype=dtype))

    if shifted and n % 2 == 0:
        grid = np.arange(-n / 2 + 1 / 2, n / 2 + 1 / 2, dtype=dtype)

    if normalized:
        if shifted and n % 2 == 0:
            grid = grid / (n / 2 - 1 / 2)
        else:
            grid = grid / (n / 2)

    x, y = np.meshgrid(grid, grid, indexing="ij")
    phi, r = cart2pol(x, y)

    return {"x": x, "y": y, "phi": phi, "r": r}


# from utils/matlab_compat.py

def m_reshape(x, new_shape):
    # This is a somewhat round-about way of saying:
    #   return x.reshape(new_shape, order='F')
    # We follow this approach since numba/cupy don't support the 'order'
    # argument, and we may want to use those decorators in the future
    # Note that flattening is required before reshaping, because
    if isinstance(new_shape, tuple):
        return m_flatten(x).reshape(new_shape[::-1]).T
    else:
        return x


def m_flatten(x):
    # This is a somewhat round-about way of saying:
    #   return x.flatten(order='F')
    # We follow this approach since numba/cupy don't support the 'order'
    # argument, and we may want to use those decorators in the future
    return x.T.flatten()
