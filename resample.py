import numpy as np
from numpy.typing import NDArray


def resample_data(
    n_pixels: int = 256,
    trim: int = None,
    resample_n_pixels: int = 121,
    data: NDArray[np.uint16] = None,
) -> NDArray[np.uint16]:
    """resamples data with default native resolution to input resampling resolution

    Args:
        n_pixels (int, optional): native pixel resolution of the detector. Defaults to 256.
        trim (int, optional): number of pixels to remove from all 4 edges. Defaults to None.
        resample_n_pixels (int, optional): number of pixels to resample too. Defaults to 121.
        data (NDArray[np.uint16], optional): input data at native resolution. Defaults to None.

    Returns:
        resampled_data (NDArray[np.uint16]): resampled array of data
    """

    if trim:
        data_trimmed = data[trim : n_pixels - trim, trim : n_pixels - trim]
    else:
        data_trimmed = data
        pass

    p = np.shape(data)[0] // resample_n_pixels
    resampled_data = np.zeros((resample_n_pixels, resample_n_pixels))

    for j in range(0, resample_n_pixels):
        for k in range(0, resample_n_pixels):
            resampled_data[j, k] = np.sum(
                data_trimmed[j * p : (j + 1) * p, k * p : (k + 1) * p]
            )

    return resampled_data
