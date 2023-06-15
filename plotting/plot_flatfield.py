import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def plot_flatfield(fname: str, signal: NDArray, distance_cm: float, det_size_cm: float):
    """
    plot a flat field of the image with a fit on it

    params:
        fname:       filename to save plot
        plot_conditions: option to plot peak finding conditions and local extrema
        condition:       'half_val' or 'quarter_val' depending on condition to check if peaks
                         are resolved
    returns:
        resolved:        true or false if the peaks are resolved based on input condition
    """
    # make x axis
    x = np.linspace(-1 * det_size_cm / 2, det_size_cm / 2, 121)
    xx = np.arctan(x / distance_cm)

    plt.plot(np.rad2deg(xx), signal, color="#F47949")
    plt.plot(np.rad2deg(xx), np.sin(np.deg2rad(90) + xx) ** (1 / 4), color="grey")

    plt.xlabel("incident angle")
    plt.ylabel("normalized intensity")

    plt.savefig(fname, dpi=300)
