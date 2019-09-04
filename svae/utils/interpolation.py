import numpy as np
from scipy.stats import norm


def lerp(start: np.ndarray,
         end: np.ndarray,
         num_steps: int,
         startpoint: bool = True,
         endpoint: bool = True) -> np.ndarray:
    """
    Linear interpolation between two points
    :param start: first point
    :param end: second point
    :param num_steps: number of interpolation steps
    :param startpoint: whether to include start point
    :param endpoint: whether to include end point
    :return: `numpy.ndarray`
    """
    start_idx = 0 if startpoint else 1
    steps = np.linspace(0, 1, num_steps, endpoint=endpoint)[start_idx:]
    steps = steps.reshape(-1, 1)
    return start * (1 - steps) + end * steps


def slerp(start: np.ndarray,
          end: np.ndarray,
          num_steps: int,
          startpoint: bool = True,
          endpoint: bool = True
          ) -> np.ndarray:
    """
    Spherical interpolation between two points
    :param start: first point
    :param end: second point
    :param num_steps: number of interpolation steps
    :param startpoint: whether to include start point
    :param endpoint: whether to include end point
    :return: `numpy.ndarray`
    """
    start_idx = 0 if startpoint else 1
    omega = np.arccos((start * end).sum() / (np.linalg.norm(start) * np.linalg.norm(end)))
    sin_omega = np.sin(omega)
    steps = np.linspace(0, 1, num_steps, endpoint=endpoint)[start_idx:]
    steps = steps.reshape(-1, 1)
    return np.sin((1.0 - steps) * omega) / sin_omega * start + np.sin(steps * omega) / sin_omega * end


def lerp_gaussian(start: np.ndarray,
                  end: np.ndarray,
                  num_steps: int,
                  startpoint: bool = True,
                  endpoint: bool = True,
                  eps: float = 1e-8):
    """
    Linear interpolation with gaussian cdf
    :param start: start point
    :param end: end point
    :param num_steps: number of interpolation steps
    :param startpoint: whether to include start point
    :param endpoint: whether to include end point
    :param eps: clipping parameter for numerical overflow prevention
    :return: `numpy.ndarray`
    """
    values = lerp(norm.cdf(start), norm.cdf(end), num_steps, startpoint, endpoint)
    return norm.ppf(np.clip(values, eps, 1 - eps))


INTERPOLATION_FUNCTIONS = {
    'lerp': lerp,
    'slerp': slerp,
    'lerp_gaussian': lerp_gaussian
}


def get_interpolation_function(interpolation_type: str):
    if interpolation_type not in INTERPOLATION_FUNCTIONS:
        raise ValueError(f"Wrong interpolation type. "
                         f"Choose one from: {INTERPOLATION_FUNCTIONS.keys()}")
    return INTERPOLATION_FUNCTIONS[interpolation_type]
