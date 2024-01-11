from simulation_engine import SimulationEngine
from hits import Hits
from plotting.plot_settings import *

import numpy as np
from numpy.typing import NDArray
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
from scipy import interpolate
from scipy.signal import peak_widths, find_peaks, convolve2d, correlate2d
import scipy.optimize as opt
from scipy.io import savemat
from skimage import color, data, restoration
import math
import cv2
import sys
from pyfftw.interfaces import scipy_fftpack as fftw
from skimage.util import random_noise
from photutils.datasets import apply_poisson_noise, make_4gaussians_image

sys.path.insert(1, "../coded_aperture_mask_designs")
from util_fncs import makeMURA, make_mosaic_MURA, get_decoder_MURA, updated_get_decoder


class Deconvolution:
    def __init__(
        self,
        hits: Hits = None,
        simulation_engine: SimulationEngine = None,
        experiment_data: NDArray[np.uint16] = None,
    ) -> None:
        if simulation_engine is not None:
            self.hits_dict = hits.hits_dict
            self.det_size_cm = simulation_engine.det_size_cm
            self.n_elements = simulation_engine.n_elements
            self.mura_elements = simulation_engine.mura_elements
            self.element_size_mm = simulation_engine.element_size_mm
            self.n_pixels = 256  # keep constant for timepix detector
            self.pixel_size = 0.0055  # cm # keep constant for timepix detector
            self.experiment = False  # TODO fix this, temporary
            self.hits_data = hits.txt_hits

            self.simulation_engine = simulation_engine
        else:
            self.experiment = True
            self.raw_heatmap = experiment_data
            self.mura_elements = 11
            self.element_size_mm = 1.21
            self.n_pixels = 256  # minpix EDU
            self.pixel_size = 0.0055  # minpix EDU
        
        self.pinhole = False
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

        # update position to shifted so origin is lower left corner (instead of center of detector)
        pos = [(pp[0] + shift, pp[1] + shift) for pp in pos_p]

        if self.trim:
            # trim the edges (remove hits that are in the trimmed portion)
            pos_trimmed = []

            for pp in pos:
                if pp[0] <= (self.trim * self.pixel_size) or pp[0] >= (
                    self.det_size_cm - (self.trim * self.pixel_size)
                ):
                    continue
                elif pp[1] <= (self.trim * self.pixel_size) or pp[1] >= (
                    self.det_size_cm - (self.trim * self.pixel_size)
                ):
                    continue
                else:
                    pos_trimmed.append(pp)

            # replace with shifted and trimmed
            self.hits_dict["Position"] = pos_trimmed
        else:
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
        if self.trim:
            # if downsampling to number of MURA elements
            heatmap, xedges, yedges = np.histogram2d(
                xxes, yxes, bins=self.resample_n_pixels
            )
            # heatmap, xedges, yedges = np.histogram2d(xxes, yxes, bins=int( (self.n_pixels-(self.trim*2))))
        elif self.pinhole:
            bin_edges_x = np.linspace(0, self.det_size_cm, int(self.downsample))
            bin_edges_y = np.linspace(0, self.det_size_cm, int(self.downsample))
            heatmap, xedges, yedges = np.histogram2d(
                xxes, yxes, bins=[bin_edges_x, bin_edges_y]
            )
        else:
            heatmap, xedges, yedges = np.histogram2d(
                xxes, yxes, bins=int(self.downsample)
            )  # just for the pinhole case - remove extra

        self.raw_heatmap = (
            heatmap  # [1:-1, 1:-1] # REMVOE THIS MOVING FORWARD ONLY FOR WEIRD STUFF
        )

        return heatmap

    def apply_dist(self, dist_type):
        """
        apply distribution to a raw heatmap
        """

        if dist_type == "sine":
            # create a sine wave
            det_dimension = np.linspace(
                -1 * self.det_size_cm / 2, self.det_size_cm / 2, self.resample_n_pixels
            )
            fov_dimension = np.arctan(det_dimension / 1)

            new_heatmap = np.zeros((self.resample_n_pixels, self.resample_n_pixels))
            for i in range(0, self.resample_n_pixels):
                row = self.raw_heatmap[i, :]
                sine_wave = np.sin(np.deg2rad(90) + fov_dimension)

                new_row = np.multiply(row, sine_wave)
                new_heatmap[:, i] = new_row

        self.raw_heatmap = new_heatmap

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
        plt.imshow(heatmap.T, origin="upper", cmap=cmap, vmax=vmax)
        plt.colorbar(label=label)
        plt.xlabel("pixel")
        plt.ylabel("pixel")
        plt.savefig(save_name, transparent=False, dpi=500)
        plt.close()

        return

    def get_mask(self, mosaic: bool = True) -> NDArray[np.uint16]:
        """
        get MURA mask design as a 2d array

        params:
        returns:
            mask: 2d numpy array of mask design, 0=hole, 1=element
        """

        # get mask design as an array of 0s and 1s
        if mosaic:
            mask, _ = make_mosaic_MURA(
                self.mura_elements,
                self.element_size_mm,
                holes=False,
                generate_files=False,
            )
        else:
            mask, dd = makeMURA(
                self.mura_elements,
                self.element_size_mm,
                holes_inv=False,
                generate_files=False,
            )
            self.dd = dd
        self.mask = mask

        return mask

    def get_decoder(self, check, rotate, delta_decoding) -> NDArray[np.uint16]:
        """
        get decoding array

        params:
        returns:
            decoder: 2d numpy array of decoding array, inverse of mask
        """

        if self.experiment:
            check = 1
        elif (
            self.mura_elements == 61
            or self.mura_elements == 67
            or self.mura_elements == 31
            or self.mura_elements == 11
            or self.mura_elements == 7
            or self.mura_elements == 73
            or self.mura_elements == 47
        ):
            check = 0
        else:
            check = 1
        decoder = updated_get_decoder(self.mura_elements)

        # resample the decoding array to the correct resolution
        decoder = np.repeat(
            np.repeat(decoder, self.resample_n_pixels // self.mura_elements, axis=1),
            self.resample_n_pixels // self.mura_elements,
            axis=0,
        )

        if delta_decoding:
            print("delta")
            decoder_zeros = np.zeros_like(decoder)
            # now replace the middle elements
            rows, cols = decoder.shape
            for row in range(1, rows, 3):
                for col in range(1, cols, 3):
                    decoder_zeros[row, col] = decoder[row, col]
            decoder = decoder_zeros

        # mismatched
        if rotate:
            self.rawIm = np.fliplr(np.flipud(self.rawIm)) # also include the flip lr for 73 ??? 
            #self.decoder = decoder * -1
            self.decoder = decoder
        else:
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
        # resizedIm = ndimage.zoom(self.rawIm, len(self.decoder) / len(self.rawIm))

        # Fourier space multiplication
        Image = np.real(
            np.fft.ifft2(np.fft.fft2(self.rawIm) * np.fft.fft2(np.rot90(self.decoder)))
        )

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
        hex_list = [
            "#0091ad",
            "#3fcdda",
            "#83f9f8",
            "#d6f6eb",
            "#fdf1d2",
            "#f8eaad",
            "#faaaae",
            "#ff57bb",
        ]
        plt.plot(self.signal, color=hex_list[-1])

        if plot_conditions:
            # plot peak finding conditions if two peaks
            b, c, half_val, quarter_val, resolved = self.check_resolved(condition)
            x_ax = np.arange(0, len(self.signal))
            plt.scatter(x_ax[b], self.signal[b], color=hex_list[1])
            plt.scatter(x_ax[c], self.signal[c], color=hex_list[0])
            plt.hlines(
                half_val,
                xmin=0,
                xmax=len(self.signal),
                linestyles="--",
                colors=hex_list[1],
            )
            plt.xlabel("pixel")
            plt.ylabel("signal")
            # plt.ylim([1e6,5e6])

            # plt.hlines(
            #    quarter_val,
            #    xmin=0,
            #    xmax=len(self.signal),
            #    linestyles="--",
            #    colors="#FF3A79",
            # )
        else:
            resolved = False

        plt.savefig(save_name, dpi=300)
        plt.close()

        return resolved

    def FWHM(self):
        peaks, _ = find_peaks(self.signal)

        # just find the width of the center peak
        # results_half = peak_widths(
        #    self.signal, [self.resample_n_pixels // 2], rel_height=0.5
        # )
        # print(results_half)
        results_half = peak_widths(self.signal, peaks, rel_height=0.5)
        return max(results_half[0])

    def flat_field(self, flat_field_array):
        """
        flat field the deconvolved signal based on pixel number
        """

        # import the flat field response (txt file - 2D array)
        # apply to each pixel
        self.flat_field_signal = np.multiply(
            self.deconvolved_image, 1 / flat_field_array
        )
        # replace
        self.deconvolved_image = self.flat_field_signal
        return

    def deconvolve(
        self,
        downsample: int = 1,
        trim: int = 0,
        plot_raw_heatmap: bool = False,
        save_raw_heatmap: str = "raw_hits_heatmap.png",
        plot_deconvolved_heatmap: bool = False,
        save_deconvolve_heatmap: str = "deconvolved_image.png",
        plot_signal_peak: bool = False,
        save_peak: str = "peak.png",
        plot_conditions: bool = False,
        check_resolved: bool = False,
        condition: str = "half_val",
        vmax: float = None,
        experiment: bool = False,
        normalize_signal: bool = False,
        apply_distribution: bool = False,
        dist_type: str = "sine",
        axis: int = 0,
        flat_field_array: NDArray = None,
        hits_txt: bool = False,
        check: int = None,
        rotate: bool = False,
        correct_collimation: bool = False,
        delta_decoding: bool = True,
        apply_noise: bool = True,
    ) -> bool:
        """
        perform all the steps to deconvolve a raw image

        params:
            downsample:               factor to reduce pixel number by (i.e. downsample = 2 means 256 becomes 128)
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

        self.downsample = downsample
        if trim:
            self.trim = trim
            self.resample_n_pixels = int(
                (self.n_pixels - (self.trim * 2)) / self.downsample
            )
        else:
            self.trim = trim
            self.resample_n_pixels = self.downsample

        # shift origin and remove unused pixels
        # data from experiment does not need to be shifted
        if not experiment and not hits_txt:
            self.shift_pos()
            self.get_raw()
            if apply_distribution:
                self.apply_dist(dist_type)
            print("SAVE TXT")
            np.savetxt(f"{save_raw_heatmap[:-3]}txt", self.raw_heatmap)
        elif hits_txt:
            self.raw_heatmap = self.hits_data

        if correct_collimation:
            self.correct_aperture_collimation(mask_thickness_um=400, focal_length_cm=2)

        if apply_noise:
            self.apply_ps_noise()

        if plot_raw_heatmap:
            self.plot_heatmap(self.raw_heatmap, save_name=save_raw_heatmap, vmax=vmax)

        self.rawIm = self.raw_heatmap

        if self.pinhole == False:
            # get mask and decoder
            self.get_mask(self.simulation_engine.mosaic)
            self.get_decoder(check, rotate, delta_decoding)

            # flip the heatmap over both axes bc point hole
            # rawIm = np.fliplr(self.raw_heatmap)

            # reflect bc correlation needs to equal convolution
            # rawIm = np.fliplr(rawIm)

            # deconvolve
            self.fft_conv()

            # self.deconvolved_image = self.deconvolved_image / (67**2 * 3**2 * 0.5)

            if flat_field_array is not None:
                self.flat_field(flat_field_array)
            if plot_deconvolved_heatmap:
                self.plot_heatmap(
                    self.deconvolved_image,
                    save_name=save_deconvolve_heatmap,
                    label="signal",
                    vmax=vmax,
                )
        """
        snr = np.amax(np.abs(self.deconvolved_image)) / np.std(
            np.abs(self.deconvolved_image)
        )

        # get max signal (point sources usually)
        # be consistent with treatment of the signal! -- it should be the AVERAGE SIGNAL AROUND SOURCE
        # max_ind = np.where(self.deconvolved_image == np.amax(self.deconvolved_image))
        max_ind = np.unravel_index(
            self.deconvolved_image.argmax(), self.deconvolved_image.shape
        )
        try:
            max_col = int(max_ind[0])
            peak_loc = int(max_ind[0])
        except:
            print("issue with getting the peak of that")
            max_col = 121
            peak_loc = 10

        self.signal = np.sum(self.deconvolved_image, axis=axis) / self.resample_n_pixels

        # normalized signal
        if normalize_signal:
            self.signal = self.signal / np.max(self.signal)
        # self.signal = np.divide(self.deconvolved_image[np.shape(self.deconvolved_image)[0]//2,:], np.max(self.deconvolved_image[np.shape(self.deconvolved_image)[0]//2,:]))
        # self.signal = self.deconvolved_image[max_col, :]
        # average of strip over the average background
        # axis 1 = ROW
        strip = int(25 / 2)

        # if max_ind[1] < 40:
        #    print(np.mean(self.deconvolved_image[60:,60:]))
        #    self.signal = (np.sum(self.deconvolved_image[max_col-strip:max_col+strip,:],axis=0)/(2*strip)) - np.mean(self.deconvolved_image[60:,60:])
        # else:
        #    self.signal = (np.sum(self.deconvolved_image[max_col-strip:max_col+strip,:],axis=0)/(2*strip)) - np.mean(np.concatenate((self.deconvolved_image[0:30,0:30],self.deconvolved_image[90:120,90:120]),axis=1))

        self.max_signal_over_noise = np.amax(self.signal) - np.mean(
            self.signal[0 : len(self.signal) // 4]
        )

        if plot_signal_peak:
            resolved = self.plot_peak(save_peak, plot_conditions, condition)
        elif check_resolved:
            _, resolved = self.check_resolved(condition)
        else:
            resolved = None
        """
        return

    def apply_ps_noise(self):
        noise_mask = np.random.poisson(self.raw_heatmap)

        self.raw_heatmap = self.raw_heatmap + noise_mask
        detector_noise = self.dark_current(self.raw_heatmap, 100/(219**2))
        self.raw_heatmap += detector_noise
        # self.raw_heatmap = random_noise(self.raw_heatmap, mode="poisson")
        # self.raw_heatmap = apply_poisson_noise(self.raw_heatmap, seed=1)

    # https://mwcraig.github.io/ccd-as-book/01-03-Construction-of-an-artificial-but-realistic-image.html

    def dark_current(
        self, image, current, exposure_time=1.0, gain=1.0, hot_pixels=False
    ):
        """
        Simulate dark current in a CCD, optionally including hot pixels.

        Parameters
        ----------

        image : numpy array
            Image whose shape the cosmic array should match.
        current : float
            Dark current, in electrons/pixel/second, which is the way manufacturers typically
            report it.
        exposure_time : float
            Length of the simulated exposure, in seconds.
        gain : float, optional
            Gain of the camera, in units of electrons/ADU.
        strength : float, optional
            Pixel count in the cosmic rays.
        """

        # dark current for every pixel; we'll modify the current for some pixels if
        # the user wants hot pixels.
        base_current = current * exposure_time / gain

        # This random number generation should change on each call.
        dark_im = np.random.poisson(base_current, size=image.shape)

        if hot_pixels:
            # We'll set 0.01% of the pixels to be hot; that is probably too high but should
            # ensure they are visible.
            y_max, x_max = dark_im.shape

            n_hot = int(0.0001 * x_max * y_max)

            # Like with the bias image, we want the hot pixels to always be in the same places
            # (at least for the same image size) but also want them to appear to be randomly
            # distributed. So we set a random number seed to ensure we always get the same thing.
            rng = np.random.RandomState(16201649)
            hot_x = rng.randint(0, x_max, size=n_hot)
            hot_y = rng.randint(0, y_max, size=n_hot)

            hot_current = 10000 * current

            dark_im[[hot_y, hot_x]] = hot_current * exposure_time / gain
        return dark_im

    def anti_mask_signal(self, save_name=None):
        self.anti_decoder = -1 * self.decoder

        # scipy.ndimage.zoom used here
        resizedIm = ndimage.zoom(self.rawIm, len(self.anti_decoder) / len(self.rawIm))
        # resizedIm = np.rot90(resizedIm)

        # Fourier space multiplication
        Image = np.real(
            np.fft.ifft2(np.fft.fft2(resizedIm) * np.fft.fft2(self.anti_decoder))
        )

        # Set minimum value to 0
        Image += np.abs(np.min(Image))

        # Shift to by half of image length after convolution
        anti_deconvolved_image = shift(
            Image, len(self.anti_decoder) // 2, len(self.anti_decoder) // 2
        )
        self.anti_deconvolved_image = anti_deconvolved_image

        self.plot_heatmap(anti_deconvolved_image, save_name=save_name)
        return self.anti_deconvolved_image

    def sum_signals(self, save_name=None):
        summed_image = self.anti_deconvolved_image + self.deconvolved_image
        self.summed_image = summed_image
        self.plot_heatmap(summed_image, save_name=save_name)
        return self.summed_image

    def plot_3D_signal(self, heatmap, heatmap2=None, save_name=None):
        x = np.arange(heatmap.shape[0])
        y = np.arange(heatmap.shape[1])
        X, Y = np.meshgrid(x, y)

        # Create a 3D figure
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Create the surface plot!
        surface = ax.plot_surface(X, Y, heatmap, cmap=cmap)
        if heatmap2 is not None:
            surface = ax.plot_surface(X, Y, heatmap2, color="r", alpha=0.2)

        # Add a color bar for reference
        fig.colorbar(surface)

        # Show the plot
        ax.set_xlabel("pixel")
        ax.set_ylabel("pixel")
        ax.set_zlabel("signal")
        # ax.set_zlim([np.amax(heatmap) / 2, np.amax(heatmap)])
        plt.show()
        plt.clf()
        # plt.savefig(save_name, dpi=300)

    def calculate_fwhm(self, direction, max_index, scale):
        def gaussian2d(xy, xo, yo, sigma_x, sigma_y, amplitude, offset):
            xo = float(xo)
            yo = float(yo)
            x, y = xy
            gg = offset + amplitude * np.exp(
                -(
                    ((x - xo) ** 2) / (2 * sigma_x**2)
                    + ((y - yo) ** 2) / (2 * sigma_y**2)
                )
            )
            return gg.ravel()

        def getFWHM_GaussianFitScaledAmp(img, direction):
            """Get FWHM(x,y) of a blob by 2D gaussian fitting
            Parameter:
                img - image as numpy array
            Returns:
                FWHMs in pixels, along x and y axes.
            """
            x = np.linspace(0, img.shape[1], img.shape[1])
            y = np.linspace(0, img.shape[0], img.shape[0])
            x, y = np.meshgrid(x, y)

            initial_guess = (img.shape[1] / 2, img.shape[0] / 2, 2, 2, 1, 0)
            params, _ = opt.curve_fit(gaussian2d, (x, y), img.ravel(), p0=initial_guess)
            xcenter, ycenter, sigmaX, sigmaY, amp, offset = params

            FWHM_x = 2 * sigmaX * np.sqrt(2 * np.log(2))
            FWHM_y = 2 * sigmaY * np.sqrt(2 * np.log(2))

            # return based on direction
            if direction == "x":
                fwhm = FWHM_y
            elif direction == "y":
                fwhm = FWHM_x
            else:
                if FWHM_x > FWHM_y:
                    fwhm = FWHM_x
                else:
                    fwhm = FWHM_y

            return fwhm, params

        # get the shifted image and scale
        img = (
            self.deconvolved_image / scale
        )  # this is the average signal over 0 after shifting for pt source centered
        import matplotlib.pyplot as plt

        sect = 6

        # get the section where the signal is
        img_bound = img.shape[0] - 1
        if max_index[0] > img_bound - sect and max_index[1] < img_bound - sect:
            img_clipped = img[
                max_index[0] - sect :, max_index[1] - sect : max_index[1] + sect + 1
            ]

            missing_rows = img_bound - max_index[0]
            missing_signal = img[
                max_index[0] - sect : max_index[0] - missing_rows,
                max_index[1] - sect : max_index[1] + sect + 1,
            ]
            flipped_pre_signal = missing_signal[::-1]
            img_clipped = np.vstack((img_clipped, flipped_pre_signal))

        elif max_index[1] > img_bound - sect and max_index[0] < img_bound - sect:
            img_clipped = img[
                max_index[0] - sect : max_index[0] + sect + 1, max_index[1] - sect :
            ]

            missing_cols = img_bound - max_index[1]
            missing_signal = img[
                max_index[0] - sect : max_index[0] + sect + 1,
                max_index[1] - sect : max_index[1] - missing_cols,
            ]
            flipped_pre_signal = missing_signal[:, ::-1]
            img_clipped = np.hstack((img_clipped, flipped_pre_signal))

        elif max_index[0] > img_bound - sect and max_index[1] > img_bound - sect:
            img_clipped = img[max_index[0] - sect :, max_index[1] - sect :]

            missing_rows = img_bound - max_index[0]
            missing_signal = img[
                max_index[0] - sect : max_index[0] - missing_rows, max_index[1] - sect :
            ]
            flipped_pre_signal = missing_signal[::-1]
            img_clipped = np.vstack((img_clipped, flipped_pre_signal))

            missing_cols = img_bound - max_index[1]
            missing_signal = img_clipped[:, : (sect - missing_cols)]
            flipped_pre_signal = missing_signal[:, ::-1]
            img_clipped = np.hstack((img_clipped, flipped_pre_signal))

        else:
            img_clipped = img[
                max_index[0] - sect : max_index[0] + sect + 1,
                max_index[1] - sect : max_index[1] + sect + 1,
            ]

        #self.plot_3D_signal(img_clipped)

        FWHM, params = getFWHM_GaussianFitScaledAmp(img_clipped, direction)

        # for plotting if interested
        xcenter, ycenter, sigmaX, sigmaY, amp, offset = params
        x = np.linspace(0, img_clipped.shape[1], 1000)
        y = np.linspace(0, img_clipped.shape[0], 1000)
        x, y = np.meshgrid(x, y)
        g2d = offset + amp * np.exp(
            -(
                ((x - xcenter) ** 2) / (2 * sigmaX**2)
                + ((y - ycenter) ** 2) / (2 * sigmaY**2)
            )
        )


        return FWHM

    def correct_aperture_collimation(self, mask_thickness_um, focal_length_cm):
        # collimation correction is a function of radial distance
        element_size_cm = self.element_size_mm * 0.1
        center_element = self.n_elements // 2

        # build the aperture collimation map - in terms of mak elements
        aperture_collimation = np.zeros((self.n_elements, self.n_elements))
        for x in range(self.n_elements):
            for y in range(self.n_elements):
                relative_x = (x - center_element) * element_size_cm
                relative_y = (y - center_element) * element_size_cm
                # calculate radial distance
                r = np.sqrt(relative_x**2 + relative_y**2)
                # solve for d
                d = mask_thickness_um * 1e-4 * (r / focal_length_cm)

                # solve for collimation factor
                c = 1 - (d / element_size_cm)
                aperture_collimation[x, y] = c

        aperture_collimation = np.real(
            np.fft.ifft2(
                np.fft.fft2(aperture_collimation)
                * np.fft.fft2(self.get_mask(mosaic=True))
            )
        )
        aperture_collimation = shift(
            aperture_collimation,
            len(self.get_mask(mosaic=True)) // 2,
            len(self.get_mask(mosaic=True)) // 2,
        )

        X, Y = np.meshgrid(
            np.arange(self.n_elements), np.arange(self.n_elements)
        )  # Creating a grid

        # Define new grid coordinates
        new_x = np.linspace(0, self.mura_elements - 1, self.mura_elements * 2)
        new_y = np.linspace(0, self.mura_elements - 1, self.mura_elements * 2)

        new_X, new_Y = np.meshgrid(new_x, new_y)  # New grid

        # Resample the grid using griddata
        new_Z = interpolate.griddata(
            (X.flatten(), Y.flatten()),
            aperture_collimation.flatten(),
            (new_X, new_Y),
            method="cubic",
        )
        # Perform the interpolation to upsample the grid
        upsampled_grid = new_Z
        plt.clf()
        plt.imshow(aperture_collimation, cmap=cmap)
        plt.colorbar()
        plt.savefig(
            "/home/rileyannereid/workspace/geant4/simulation-results/rings/aperture_collimation.png"
        )
        plt.clf()

        # now correct the image
        self.raw_heatmap = np.divide(self.raw_heatmap[:-1, :-1], aperture_collimation)

    def shift_noise_floor_ptsrc(self, row, col):
        # shift the image down to the noise floor
        # exclude the banding artifacts
        noise_floor = np.ones_like(self.deconvolved_image, dtype=bool)
        noise_floor[row - 1 : row + 2, :] = False
        noise_floor[:, col - 1 : col + 2] = False
        average_noise = np.mean(self.deconvolved_image[noise_floor])

        self.shifted_image = self.deconvolved_image - average_noise
        return self.shifted_image

    def lucy_richardson(self, save_name):
        # grab the middle of the mask pattern if mosiacked

        lr_mask = self.mask[
            self.mura_elements // 2 : (self.mura_elements // 2) + self.mura_elements,
            self.mura_elements // 2 : (self.mura_elements // 2) + self.mura_elements,
        ]
        lr_mask = np.repeat(
            np.repeat(lr_mask, self.resample_n_pixels // self.mura_elements, axis=1),
            self.resample_n_pixels // self.mura_elements,
            axis=0,
        )
        self.lr_mask = lr_mask

        psf = convolve2d(self.decoder, lr_mask, "same")
        lr_image = restoration.richardson_lucy(
            self.deconvolved_image, -1 * psf, num_iter=1
        )

        self.lr_image = lr_image

        self.plot_heatmap(lr_image, save_name)

    def reverse_raytrace(self):
        # attempt to reverse ray trace each pixel through each coded aperture hole
        # generate statistical map of hits

        # need locations of each coded apeture hole
        cell_size_cm = 0.05
        px_size_cm = 0.025
        focal_length = 2  # cm
        src_plane = 3  # cm

        mask_center = 30
        # detector_center = 30
        detector_center = 60.5

        # raw_hits = zoom(self.raw_heatmap, 0.5)
        raw_hits = self.raw_heatmap

        # mask is centered at 0,0, focal length
        mask_coordinates = []
        mask = self.get_mask(mosaic=False)
        for row_idx, row in enumerate(mask):
            for col_idx, value in enumerate(row):
                if value == 0:
                    x = (col_idx - mask_center) * cell_size_cm
                    y = (row_idx - mask_center) * cell_size_cm
                    mask_coordinates.append((x, y))

        # detector is centered at 0,0,0 - where 0,0 is the corner of the center 4 pixels
        px_coordinates = []
        for row_idx, row in enumerate(raw_hits):
            for col_idx, value in enumerate(row):
                x = (col_idx - detector_center) * px_size_cm
                y = (row_idx - detector_center) * px_size_cm
                px_coordinates.append((x, y))

        # lol so this whole thing actually simplifies to
        t = src_plane / focal_length

        # use t to find the point on the source plane LOL
        # new_x = px[0] + (mx[0] - px[0]) * t
        # new_y = px[1] + (mx[1] - px[1]) * t
        # new_z = src_plane

        # build the matrix
        mx = [arr[0] for arr in mask_coordinates]
        mx = np.vstack(mx)

        my = [arr[1] for arr in mask_coordinates]
        my = np.vstack(my)

        xes = []
        yes = []
        existing_histogram = np.zeros((122, 122))
        x_edges = np.linspace(-2.75, 2.75, 122 + 1)
        y_edges = np.linspace(-2.75, 2.75, 122 + 1)

        for ii, pxy in enumerate(px_coordinates):
            new_x = pxy[0] * (1 - t) + t * mx
            new_y = pxy[1] * (1 - t) + t * my
            new_z = src_plane

            # invert to get hits
            col_idx = int(pxy[0] / px_size_cm)
            row_idx = int(pxy[1] / px_size_cm)

            px_hits = raw_hits[row_idx, col_idx]

            histogram, _, _ = np.histogram2d(
                new_x.flatten(), new_y.flatten(), bins=[x_edges, y_edges]
            )

            if px_hits > 0:
                existing_histogram += histogram * int(px_hits)
            else:
                pass

        plt.clf()
        plt.imshow(existing_histogram)
        plt.savefig("test.png")
        return mx, my

    def export_to_matlab(self, openelements=None):
        # Define a dictionary to store the grid
        no_mosaic_mask = self.lr_mask
        no_mosaic_decoder = self.decoder
        data = {"mask": no_mosaic_mask}

        # Specify the file path where you want to save the .mat file
        file_path = "mask.mat"

        # Save the grid data as a .mat file
        savemat(file_path, data)

        # save resampled hits as well

        data = {"hits": self.raw_heatmap}

        # Specify the file path where you want to save the .mat file
        file_path = "hits.mat"

        # Save the grid data as a .mat file
        savemat(file_path, data)

        data = {"decoder": no_mosaic_decoder}

        # Specify the file path where you want to save the .mat file
        file_path = "decoder.mat"

        # Save the grid data as a .mat file
        savemat(file_path, data)

        # data = {"openElements": openelements}

        # Specify the file path where you want to save the .mat file
        # file_path = "open_elements.mat"

        # Save the grid data as a .mat file
        # savemat(file_path, data)

        return

    """ # TODO! move these to a new script 
    def plot_flux_signal(self, ax, simulation_engine, fname):
        # calculate incident flux assuming isotropic
        j = simulation_engine.n_particles / (
            4 * np.pi**2 * simulation_engine.radius_cm**2
        )
        # units are [cm^-2 s^-1 sr^-1]
        counts = np.sum(self.raw_heatmap)
        geom_factor = counts / j

        flux_inst = self.raw_heatmap * geom_factor
        pad = np.sum(flux_inst, 1)
        #plt.clf()

        # need to convert x axis from pixels to angle
        x_axis = np.linspace(0, len(pad) - 1, len(pad)) - len(pad) / 2
        angle = np.rad2deg(
            np.arctan((x_axis * self.pixel_size) / (simulation_engine.mask_gap_cm * 10))
        )

        ax.scatter(angle, pad, color="#A2E3C4")
        plt.errorbar(
            angle,
            pad,
            xerr=np.rad2deg(
                np.arctan(
                    (simulation_engine.element_size_mm)
                    / (simulation_engine.mask_gap_cm * 10)
                )
            ),
            color="#A2E3C4",
            fmt="o",
        )

        ax = plt.gca()
        ax.set_yscale("log")
        ax.set_ylim([1, 1e4])
        ax.set_ylabel("flux [/cm^2 s sr]")
        ax.set_xlabel("polar angle")
        #plt.show()
        #plt.savefig(fname)


    def plot_signal_on_distribution(self, fov_deg, save_name="sine_comparison"):

        # get max signal
        self.signal = np.sum(self.raw_heatmap, axis=0)
        max_signal = np.amax(self.signal)

        # rescale x axis to fov
        xx = np.radians(fov_deg) * np.arange(0, self.multiplier) / self.multiplier
        xx = [x + np.radians(90 - (fov_deg / 2)) for x in xx]

        # plot normalized signal
        plt.plot(xx, self.signal / max_signal, "#EA526F", label="signal")

        # plot distribution -- sine
        time = np.arange(0, np.pi, 0.01)
        # amplitude of the sine wave is sine of a variable like time
        amplitude = np.sin(time)
        plt.plot(time, amplitude, "#070600", label="sine dist.")
        plt.fill_between(
            time,
            amplitude,
            0,
            where=(time >= np.radians(90 - (fov_deg / 2)))
            & (time < np.radians(90 + (fov_deg / 2))),
            color="#23B5D3",
            alpha=0.4,
            label="FOV",
        )

        # plot
        plt.legend()
        plt.xlim([0, np.pi])
        plt.ylim([0, 1.1])
        plt.savefig("../results/pinhole/%s.png" % (save_name), dpi=300)
        plt.close()
    """


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
