import numpy as np


def lerp(start: np.ndarray, end: np.ndarray, num_steps: int) -> np.ndarray:
    """
    Linear interpolation between two points
    :param start: first point
    :param end: second point
    :param num_steps: number of interpolation steps
    :return: `numpy.ndarray`
    """
    steps = np.linspace(0, 1, num_steps, endpoint=False)[1:]
    steps = steps.reshape(-1, 1)
    return start * (1 - steps) + end * steps


def slerp(start: np.ndarray, end: np.ndarray, num_steps: int) -> np.ndarray:
    """
    Spherical interpolation between two points
    :param start:
    :param end:
    :param num_steps:
    :return:
    """
    omega = np.arccos((start * end).sum() / (np.linalg.norm(start) * np.linalg.norm(end)))
    sin_omega = np.sin(omega)
    steps = np.linspace(0, 1, num_steps, endpoint=False)[1:]
    steps = steps.reshape(-1, 1)
    return np.sin((1.0 - steps) * omega) / sin_omega * start + np.sin(steps * omega) / sin_omega * end
