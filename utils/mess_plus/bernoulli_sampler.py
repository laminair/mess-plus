import numpy as np

from numpy.random import binomial


def sample_from_bernoulli(c: float, timestamp: int):
    p_t = min(
        1.0, c / np.power(1 if timestamp == 0 else timestamp, (1 / 5))
    )

    x_t = binomial(n=1, p=p_t, size=1)

    return p_t, x_t.item()
