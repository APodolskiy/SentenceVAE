from math import log, exp


class Annealing:
    def __init__(self,
                 max_value: float = 1.0,
                 steps: int = 5000,
                 warm_up_steps: int = 0):
        """
        Base class for annealing functions.
        Basically one should implement only _func method.
        :param max_value: maximum value function value
        :param steps: number of steps to acquire maximum value
        :param warm_up_steps: number of warm up steps when function returns zero.
        """
        self.max_value = max_value
        self.steps = steps
        self.warm_up_steps = warm_up_steps
        self.num_steps = 0

    def __call__(self) -> float:
        """
        Perform a step on annealing function.
        :return: annealing function value
        """
        step = self.num_steps
        self.num_steps += 1
        if step < self.warm_up_steps:
            return 0
        return min(self.max_value, self._func(step - self.warm_up_steps))

    def _func(self, x: int) -> float:
        """
        Annealing update function.
        :param x: function argument
        :return: function value
        """
        raise NotImplemented("Update function is not implemented!")


class LinearAnnealing(Annealing):
    """
    Linear annealing function
    """
    def _func(self, x: int) -> float:
        return self.max_value * x / self.steps


class LogisticAnnealing(Annealing):
    def __init__(self,
                 max_value: float = 1.0,
                 steps: int = 5000,
                 warm_up_steps: int = 0,
                 fast: bool = False,
                 eps: float = 1e-5):
        """
        Sigmoid annealing function.
        :param max_value: maximum value function value
        :param steps: number of steps to acquire maximum value
        :param warm_up_steps: number of warm up steps when function returns zero.
        :param fast: use fast approximation of sigmoid function
        """
        super(LogisticAnnealing, self).__init__(max_value=max_value,
                                                steps=steps,
                                                warm_up_steps=warm_up_steps)
        self.k = -(log(-1 + 1/(1 - eps)))/(0.5 * self.steps)
        self.fast = fast

    def _func(self, x: int) -> float:
        if self.fast:
            return 0.5*self.k*(x - 0.5*self.steps) / (1 + self.k*abs(x - 0.5*self.steps)) + 0.5
        return 1 / (1 + exp(-self.k * (x - 0.5*self.steps)))
