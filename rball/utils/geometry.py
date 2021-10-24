import numpy as np
import numba as nb


@nb.njit(fastmath=True)
def ang2cart(ra: float, dec: float) -> np.ndarray:
    """
    convert spherical coordinates (in deg)
    to cartesian

    :param ra:
    :param dec:
    :return:
    """
    pos = np.zeros(3)
    ra = np.deg2rad(ra)
    dec = np.deg2rad(dec)

    pos[0] = np.cos(dec) * np.cos(ra)
    pos[1] = np.cos(dec) * np.sin(ra)
    pos[2] = np.sin(dec)

    return pos


@nb.njit(fastmath=True)
def get_ang(X1: np.ndarray, X2: np.ndarray) -> float:
    """
    get the angular seperation (in radian)
    between two cartesian vectors

    :param X1:
    :param X2:
    :return:
    """
    norm1 = np.sqrt(X1.dot(X1))
    norm2 = np.sqrt(X2.dot(X2))
    tmp = np.clip(np.dot(X1 / norm1, X2 / norm2), -1, 1)

    return np.arccos(tmp)


@nb.njit(fastmath=True)
def is_occulted(ra: float, dec: float, sc_pos: np.ndarray):
    """
    compute if the earth shadows the

    :param ra:
    :param dec:
    :param sc_pos:
    :return:
    """
    # min_vis = 1.1955505376161157  # 68.5

    earth_radius = 6371.0
    spacecraft_radius = np.sqrt((sc_pos ** 2).sum())

    horizon_angle = 90 - np.rad2deg(np.arccos(earth_radius / spacecraft_radius))

    min_vis = np.deg2rad(horizon_angle)

    cart_position = ang2cart(ra, dec)
    ang_sep = get_ang(cart_position, -sc_pos)

    return ang_sep < min_vis
