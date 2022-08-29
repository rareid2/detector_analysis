from simulation_engine import SimulationEngine
from hits import Hits

import numpy as np
from numpy.typing import NDArray
from typing import Tuple
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import sys

sys.path.insert(1, "../coded_aperture_mask_designs")
from util_fncs import makeMURA, make_mosaic_MURA, get_decoder_MURA


class Deconvolution:
    def __init__(self, hits: Hits, simulation_engine: SimulationEngine) -> None:

        self.hits_dict = hits.hits_dict
        self.det_size_cm = simulation_engine.det_size_cm
        self.n_elements = simulation_engine.n_elements
        self.mura_elements = simulation_engine.mura_elements
        self.element_size_mm = simulation_engine.element_size_mm

        return

    def shift_pos(self) -> None:
        """
        shift positions on detector so that origin is lower left corner

        params:
        returns:
        """

        # set the size to shift position
        pos_p = self.hits_dict["Position"]
        shift = self.det_size_cm / 2

        # update position to shifted so origin is lower left corner
        pos = [(pp[0] + shift, pp[1] + shift) for pp in pos_p]
        self.hits_dict["Position"] = pos

        return

    def get_raw(self) -> NDArray[np.uint16]:
        """
        get raw hits and 2d histogram

        params:
        returns:
            heatmap: 2d histogram of hits
        """

        xxes = [p[0] for p in self.hits_dict["Position"]]
        yxes = [p[1] for p in self.hits_dict["Position"]]

        # get heatmap
        heatmap, xedges, yedges = np.histogram2d(
            xxes, yxes, bins=self.multiplier * self.mura_elements
        )

        self.raw_heatmap = heatmap

        return heatmap

    def plot_heatmap(
        self,
        heatmap: NDArray[np.uint16],
        save_name: str,
        label: str = "# particles",
        vmax: float = None,
    ) -> None:
        """
        plot a heatmap using imshow

        params:
            heatmap:   numpy 2d array to plot
            save_name: file_name to save under
            label:     label for colorbar
            vmax:      max for colorbar
        returns:
        """

        # plot transpose for visualization
        plt.imshow(heatmap.T, origin="lower", cmap="RdBu_r", vmax=vmax)
        plt.colorbar(label=label)
        plt.savefig(save_name, dpi=300)
        plt.close()

        return

    def get_mask(self) -> NDArray[np.uint16]:
        """
        get MURA mask design as a 2d array

        params:
        returns:
            mask: 2d numpy array of mask design, 0=hole, 1=element
        """

        # get mask design as an array of 0s and 1s
        mask, _ = make_mosaic_MURA(
            self.mura_elements, self.element_size_mm, holes=False, generate_files=False
        )
        self.mask = mask

        return mask

    def get_decoder(self) -> NDArray[np.uint16]:
        """
        get decoding array

        params:
        returns:
            decoder: 2d numpy array of decoding array, inverse of mask
        """

        if self.mura_elements == 67 or self.mura_elements == 31:
            check = 0
        else:
            check = 1

        decoder = get_decoder_MURA(
            self.mask, self.mura_elements, holes_inv=False, check=check
        )
        decoder = np.repeat(decoder, self.multiplier, axis=1).repeat(
            self.multiplier, axis=0
        )

        self.decoder = decoder

        return decoder

    def fft_conv(self) -> NDArray[np.uint16]:
        """
        perform deconvolution using inverse fft

        params:
        returns:
            deconvolved_image: 2d numpy array of resulting image from deconvolution
        """

        # scipy.ndimage.zoom used here
        resizedIm = zoom(self.rawIm, len(self.decoder) / len(self.rawIm))

        # Fourier space multiplication
        Image = np.real(
            np.fft.ifft2(np.fft.fft2(resizedIm) * np.fft.fft2(self.decoder))
        )

        # Set minimum value to 0
        Image += np.abs(np.min(Image))

        # Shift to by half of image length after convolution
        deconvolved_image = shift(Image, len(self.decoder) // 2, len(self.decoder) // 2)
        self.deconvolved_image = deconvolved_image

        return deconvolved_image

    def check_resolved(
        self, condition: str
    ) -> Tuple[NDArray[np.uint16], NDArray[np.uint16], float, float, bool]:
        """
        check if two peaks are "resolved", i.e. the valley between them is lower than
        the peak by atleast half/quarter of the peak height above the noise floor

        params:
            condition:   'half_val' or 'quarter_val' depending on condition to check if peaks
                         are resolved
        returns:
            b:           array with indices of local minima
            c:           array with indices of local maxima
            half_val:    value that is peak height above noise - half peak height
            quarter_val: value that is peak height above noise - quarter of peak height above noise
            resolved:    true or false if the peaks are resolved based on input condition
        """

        # find local min and maxes
        b = (np.diff(np.sign(np.diff(self.signal))) > 0).nonzero()[0] + 1
        c = (np.diff(np.sign(np.diff(self.signal))) < 0).nonzero()[0] + 1

        # find the two peaks in the signal
        local_maxes = self.signal[c]
        local_maxes.sort()
        largest_peak = local_maxes[-1]
        second_largest_peak = local_maxes[-2]
        largest_local_min = np.max(self.signal[b])

        # define conditions for half and quarter separated
        half_val = (
            np.max(self.signal) - np.mean(self.signal[0 : len(self.signal) // 4])
        ) // 2
        half_val = half_val + np.mean(self.signal[0 : len(self.signal) // 4])
        quarter_val = (
            np.max(self.signal) - np.mean(self.signal[0 : len(self.signal) // 4])
        ) // 4
        quarter_val = 3 * quarter_val + np.mean(self.signal[0 : len(self.signal) // 4])

        if condition == "half_val":
            # first, are there two peaks?
            # peak here is defined as larger than half value
            if largest_peak > half_val and second_largest_peak > half_val:
                # is the largest local min below the condition
                if largest_local_min < half_val:
                    resolved = True
                else:
                    resolved = False
                    # print('peaks are separated but not enough')
            else:
                resolved = False
                # print('peaks are not separated')

        elif condition == "quarter_val":
            if largest_peak > quarter_val and second_largest_peak > quarter_val:
                # is the largest local min below the condition
                if largest_local_min < quarter_val:
                    resolved = True
                else:
                    resolved = False
            else:
                resolved = False

        return b, c, half_val, quarter_val, resolved

    def plot_peak(self, save_name: str, plot_conditions: bool, condition: str) -> bool:
        """
        plot a slice of the deconvolved image to see signal strength over noise floor

        params:
            save_name:       filename to save plot
            plot_conditions: option to plot peak finding conditions and local extrema
            condition:       'half_val' or 'quarter_val' depending on condition to check if peaks
                             are resolved
        returns:
            resolved:        true or false if the peaks are resolved based on input condition
        """

        # plot the signal
        plt.plot(self.signal, color="#34C0D2")

        if plot_conditions:
            # plot peak finding conditions if two peaks
            b, c, half_val, quarter_val, resolved = self.check_resolved(condition)
            x_ax = np.arange(0, len(self.signal))
            plt.scatter(x_ax[b], self.signal[b], color="b")
            plt.scatter(x_ax[c], self.signal[c], color="r")
            plt.hlines(
                half_val,
                xmin=0,
                xmax=len(self.signal),
                linestyles="--",
                colors="#FF3A79",
            )
            plt.hlines(
                quarter_val,
                xmin=0,
                xmax=len(self.signal),
                linestyles="--",
                colors="#FF3A79",
            )
        else:
            resolved = False

        plt.savefig(save_name, dpi=300)
        plt.close()

        return resolved

    def deconvolve(
        self,
        multiplier: int,
        plot_raw_heatmap: False = bool,
        save_raw_heatmap: str = "raw_hits_heatmap.png",
        plot_deconvolved_heatmap: bool = False,
        save_deconvolve_heatmap: str = "deconvolved_image.png",
        plot_signal_peak: bool = False,
        save_peak: str = "peak.png",
        plot_conditions: bool = False,
        check_resolved: bool = False,
        condition: str = "half_val",
        vmax: float = None,
    ) -> bool:
        """
        perform all the steps to deconvolve a raw image

        params:
            multiplier:               ??????
            plot_raw_heatmap:         option to plot a heatmap of raw signal
            save_raw_heatmap:         name to save figure as for raw image
            plot_deconvolved_heatmap: option to plot a heatmap of deconvolved signal
            save_deconvolve_heatmap:  name to save figure as for deconvolved image
            plot_signal_peak:         option to plot a slice along the deconvolved signal where peaks are
            save_peak:                name to save figure as for peaks
            plot_conditions:          option to plot peak finding conditions and local extrema
            check_resolved:           check if two peaks are resolved or not based on input condition
            condition:                'half_val' or 'quarter_val' depending on condition to check if peaks
                                      are resolved
            vmax:                     max for colorbar
        returns:
            resolved:                 true or false if the peaks are resolved based on input condition
        """

        self.multiplier = multiplier

        # shift origin
        self.shift_pos()

        # get heatmap
        self.get_raw()

        if plot_raw_heatmap:
            self.plot_heatmap(self.raw_heatmap, save_name=save_raw_heatmap, vmax=vmax)

        # get mask and decoder
        self.get_mask()
        self.get_decoder()

        # flip the heatmap over both axes bc point hole
        rawIm = np.fliplr(np.flipud(self.raw_heatmap))

        # reflect bc correlation needs to equal convolution
        rawIm = np.fliplr(rawIm)
        self.rawIm = rawIm

        # deconvolve
        self.fft_conv()
        if plot_deconvolved_heatmap:
            self.plot_heatmap(
                self.deconvolved_image,
                save_name=save_deconvolve_heatmap,
                label="signal",
                vmax=vmax,
            )

        snr = np.amax(np.abs(self.deconvolved_image)) / np.std(
            np.abs(self.deconvolved_image)
        )

        # get max signal (point sources usually)
        max_ind = np.where(self.deconvolved_image == np.amax(self.deconvolved_image))
        max_col = max_ind[1]
        if np.shape(max_col)[0] > 1:
            max_col = max_col[0]

        # self.signal = np.fliplr(self.deconvolved_image)[:, int(max_col)]
        # self.signal = np.fliplr(self.deconvolved_image)[:, 536]
        self.signal = np.sum(self.deconvolved_image, axis=0)

        if plot_signal_peak:
            resolved = self.plot_peak(save_peak, plot_conditions, condition)
        elif check_resolved:
            _, resolved = self.check_resolved(condition)
        else:
            resolved = None

        return resolved


def shift(m, hs, vs):
    """
    i dont know what this does

    params:
        m:  input image
        hs: horizontal shift
        vs: vertical shift
    returns:
    """

    hs += 1
    vs += 1

    # Get original image size
    rm, cm = np.shape(m)

    # Shift each quadrant by amount [hs, vs]
    m = np.block(
        [
            [m[rm - vs : rm, cm - hs : cm], m[rm - vs : rm, 0 : cm - hs]],
            [m[0 : rm - vs, cm - hs : cm], m[0 : rm - vs, 0 : cm - hs]],
        ]
    )

    return m
