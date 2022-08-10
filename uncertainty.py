import numpy as np
from typing import Tuple


def add_uncertainty(
    pos: Tuple, dist_type: str, dist_param: float, bound: float
) -> Tuple:
    """
    generates random sample and updates position to include uncertainty

    params:
        pos:        xy positions on the detector
        dist_type:  "Gaussian", "Poission", or "Uniform" distribution to sample uncertainty from
        dist_param: determined by dist_type, in milimeters
                    if "Gaussian" standard deviation
                    if "Poission" 1/lambda
                    if "Uniform" bounds for uniform
        bound:      dimensions of detector (cm) to make sure uncertainty doesnt push off detector
    returns:
        newpos:     new position with uncertainty added
    """

    # confirm within bounds
    while True:

        # generate values in polar coordinates
        if dist_type == "Gaussian":
            mu = 0
            r = np.random.normal(mu, dist_param)
        elif dist_type == "Poisson":
            r = np.random.poisson(1 / dist_param)
        elif dist_type == "Uniform":
            r = np.random.random() * (2 * dist_param) - dist_param
        else:
            raise ValueError("enter a valid distribution for position uncertainty")

        # generate a random azimuth
        azimuth = np.random.random() * 2 * np.pi

        # perturb original position in cm transforming back to cartesian
        dx = (r / 10) * np.cos(azimuth)
        dy = (r / 10) * np.sin(azimuth)

        # update positions
        posx = pos[0] + dx
        posy = pos[1] + dy

        # confirm the new position is within the bounds of the detector
        if np.abs(posx) <= bound and np.abs(posy) <= bound:
            break

    newpos = (posx, posy)

    return newpos
