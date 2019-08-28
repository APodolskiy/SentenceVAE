import numpy as np


def lerp(p1: np.ndarray, p2: np.ndarray, num_steps: int):
    """
    Linear interpolation between two points
    :param p1: first point
    :param p2: second point
    :param num_steps: number of interpolation steps
    :return: `numpy.ndarray`
    """
    for step in np.linspace(0, 1, num_steps, endpoint=False)[1:]:
        yield p1 * (1 - step) + p2 * step


def slerp(p1: np.ndarray, p2: np.ndarray, num_steps: int):
    """
    Spherical interpolation between two points
    :param p1:
    :param p2:
    :param num_steps:
    :return:
    """
    omega = np.arccos(np.dot(p1 / np.linalg.norm(p2), p2 / np.linalg.norm(p2)))
    sin_omega = np.sin(omega)
    for step in np.linspace(0, 1, num_steps, endpoint=False)[1:]:
        yield np.sin((1.0 - step) * omega) / sin_omega * p1 + np.sin(step * omega) / sin_omega * p2
