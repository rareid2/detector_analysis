import numpy as np


def add_uncertainty(pos, dist_type, dist_param, bound):
    """
    generates random sample and updates position to include uncertainty

    :param pos: tuple with (x,y) position
    :param dist_type: string, "Gaussian" or "Poission" or "Uniform"
    :param dist_param: float, determined by dist_type, in milimeters
        if "Gaussian" standard deviation
        if "Poission" 1/lambda
        if "Uniform" bounds for uniform
    :param bound: boundary of detector

    :return: tuple with updated (x,y) position
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

    return (posx, posy)
