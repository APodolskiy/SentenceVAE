import numpy as np


def lerp(p1: np.ndarray, p2: np.ndarray, num_steps: int) -> np.ndarray:
    """
    Linear interpolation between two points
    :param p1: first point
    :param p2: second point
    :param num_steps: number of interpolation steps
    :return: `numpy.ndarray`
    """
    steps = np.linspace(0, 1, num_steps, endpoint=False)[1:]
    steps = steps.reshape(-1, 1)
    return p1 * (1 - steps) + p2 * steps


def slerp(p1: np.ndarray, p2: np.ndarray, num_steps: int) -> np.ndarray:
    """
    Spherical interpolation between two points
    :param p1:
    :param p2:
    :param num_steps:
    :return:
    """
    omega = np.arccos((p1 * p2).sum() / (np.linalg.norm(p1) * np.linalg.norm(p2)))
    sin_omega = np.sin(omega)
    steps = np.linspace(0, 1, num_steps, endpoint=False)[1:]
    steps = steps.reshape(-1, 1)
    return np.sin((1.0 - steps) * omega) / sin_omega * p1 + np.sin(steps * omega) / sin_omega * p2
