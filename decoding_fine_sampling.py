import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# colormap
colors = [(0, 0, 0), (255, 255, 255)]
cmap_name = "01"
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=2)


# -------------- -------------- -------------- -------------- --------------
def fine_sampling_decoding(
    decoder: NDArray[np.uint16], alpha: int = 3, plot: bool = False
) -> NDArray[np.uint16]:
    # standard decoding array - repeat the values to match new size

    num_axes = decoder.ndim

    if num_axes == 2:
        decoder = np.repeat(
            np.repeat(decoder, alpha, axis=1),
            alpha,
            axis=0,
        )
    elif num_axes == 1:
        decoder = np.repeat(decoder, alpha)

    else:
        raise ValueError("Decoder is not a 1D or 2D array")

    # plotting
    if plot:
        if num_axes == 1:
            # reshape just for plotting
            plt.imshow(decoder.reshape(1, -1), cmap=cmap)
        else:
            plt.imshow(decoder, cmap=cmap)
        plt.colorbar()
        plt.title(rf"delta decoding with fine sampling $\alpha$ =  {alpha} ")
        plt.show()

    return decoder


# -------------- -------------- -------------- -------------- --------------
def delta_decoding(
    decoder: NDArray[np.uint16], alpha: int = 3, plot: bool = False
) -> NDArray[np.uint16]:
    # delta decoding - extra pixels are replaced with 0s

    num_axes = decoder.ndim

    if num_axes == 2:
        decoder = np.repeat(
            np.repeat(decoder, alpha, axis=1),
            alpha,
            axis=0,
        )

        # start with all 0s
        decoder_zeros = np.zeros_like(decoder)

        # now replace the middle elements to be 1
        middle_index = alpha // 2

        rows, cols = decoder.shape

        # iterate by alpha
        for row in range(middle_index, rows, alpha):
            for col in range(middle_index, cols, alpha):
                decoder_zeros[row, col] = decoder[row, col]

    elif num_axes == 1:
        decoder = np.repeat(decoder, alpha)

        # start with all 0s
        decoder_zeros = np.zeros_like(decoder)

        # now replace the middle elements to be 1
        middle_index = alpha // 2

        array_len = len(decoder)

        for ar in range(middle_index, array_len, alpha):
            decoder_zeros[ar] = decoder[ar]

    else:
        raise ValueError("Decoder is not a 1D or 2D array")

    # new delta decoder
    decoder = decoder_zeros

    # plotting
    if plot:
        if num_axes == 1:
            # reshape just for plotting
            plt.imshow(decoder.reshape(1, -1), cmap=cmap)
        else:
            plt.imshow(decoder, cmap=cmap)
        plt.colorbar()
        plt.title(rf"delta decoding with fine sampling $\alpha$ =  {alpha} ")
        plt.show()

    return decoder


# -------------- -------------- -------------- -------------- --------------
# assuming you start with some decoding array called decoder
# decoder is the size of the coded aperture mask array

# for now we can test with something random

size = 31

# random array of 0s and 1s
decoder = np.random.randint(2, size=(size, size))

# plotting
if decoder.ndim == 1:
    # reshape just for plotting
    plt.imshow(decoder.reshape(1, -1), cmap=cmap)
    plt.colorbar()
    plt.title("decoding with no fine sampling")
    plt.show()
elif decoder.ndim == 2:
    plt.imshow(decoder, cmap=cmap)
    plt.colorbar()
    plt.title("decoding with no fine sampling")
    plt.show()
else:
    print("cannot display 0d or > 2d decoder")


# alpha determines the amount of fine sampling
# for best reconstruction choose an integer value of alpha
alpha = 3

try:
    delta_decoding(decoder, alpha, plot=True)
except ValueError as e:
    print(f"Error: {e}")
