from simulation_engine import SimulationEngine
from hits import Hits
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

import sys
sys.path.insert(1, "../coded_aperture_mask_designs")
from util_fncs import makeMURA, make_mosaic_MURA, get_decoder_MURA

class Deconvolution:
    def __init__(self, hits: Hits, simulation_engine: SimulationEngine):
        self.hits_dict = hits.hits_dict
        self.det_size_cm = simulation_engine.det_size_cm
        self.n_elements = simulation_engine.n_elements
        self.mura_elements = simulation_engine.mura_elements
        self.element_size_mm = simulation_engine.element_size_mm

    def shift_pos(self):
        # set the size to shift position
        pos_p = self.hits_dict["Position"]
        shift = self.det_size_cm / 2

        # update position to shifted so origin is lower left corner
        pos = [(pp[0] + shift, pp[1] + shift) for pp in pos_p]
        self.hits_dict["Position"] = pos

    def get_raw(self):
        xxes = [p[0] for p in self.hits_dict["Position"]]
        yxes = [p[1] for p in self.hits_dict["Position"]]

        # plot heatmap
        heatmap, xedges, yedges = np.histogram2d(
            xxes, yxes, bins=self.multiplier * self.mura_elements
        )

        self.raw_heatmap = heatmap
        return heatmap
    
    def plot_heatmap(self,heatmap,save_name,label='# particles'):

        plt.imshow(heatmap.T, origin="lower", cmap="RdBu_r")
        plt.colorbar(label=label)
        plt.savefig(save_name, dpi=300)
        plt.close()
    
    def get_mask(self):

        mask, decode = make_mosaic_MURA(
                self.mura_elements, self.element_size_mm, holes=False, generate_files=False
            )
        self.mask = mask
        return mask

    def get_decoder(self):
        
        if self.mura_elements == 67 or self.mura_elements == 31:
            check = 0
        else:
            check = 1

        decoder = get_decoder_MURA(self.mask, self.mura_elements, holes_inv=False, check=check)
        decoder = np.repeat(decoder, self.multiplier, axis=1).repeat(self.multiplier, axis=0)

        self.decoder = decoder
        return decoder

    def fft_conv(self):

        # scipy.ndimage.zoom used here
        resizedIm = zoom(self.rawIm, len(self.decoder) / len(self.rawIm))

        # Fourier space multiplication
        Image = np.real(np.fft.ifft2(np.fft.fft2(resizedIm) * np.fft.fft2(self.decoder)))

        # Set minimum value to 0
        Image += np.abs(np.min(Image))

        # Shift to by half of image length after convolution
        deconvolved_image = shift(Image, len(self.decoder) // 2, len(self.decoder) // 2)
        self.deconvolved_image = deconvolved_image
        return deconvolved_image

    def check_resolved(self,condition):
        b = (np.diff(np.sign(np.diff(self.signal))) > 0).nonzero()[0] + 1
        c = (np.diff(np.sign(np.diff(self.signal))) < 0).nonzero()[0] + 1

        local_maxes = self.signal[c]
        local_maxes.sort()
        largest_peak = local_maxes[-1]
        second_largest_peak = local_maxes[-2]
        largest_local_min = np.max(self.signal[b])

        # define conditions for half and quarter separated
        half_val = (np.max(self.signal) - np.mean(self.signal[0 : len(self.signal) // 4])) // 2
        half_val = half_val + np.mean(self.signal[0 : len(self.signal) // 4])
        quarter_val = (np.max(self.signal) - np.mean(self.signal[0 : len(self.signal) // 4])) // 4
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

        return b,c,half_val,quarter_val,resolved


    def plot_peak(self,save_name,plot_conditions,condition):
        # plot the signal
        plt.plot(self.signal, color="#34C0D2")

        if plot_conditions:
            b,c,half_val,quarter_val,resolved = self.check_resolved(condition)
            x_ax = np.arange(0, len(self.signal))
            plt.scatter(x_ax[b], self.signal[b], color="b")
            plt.scatter(x_ax[c], self.signal[c], color="r")
            plt.hlines(half_val, xmin=0, xmax=len(self.signal), linestyles="--", colors="#FF3A79")
            plt.hlines(quarter_val, xmin=0, xmax=len(self.signal), linestyles="--", colors="#FF3A79")
        else:
            resolved = None

        plt.savefig(save_name, dpi=300)
        plt.close()

        return resolved

    def deconvolve(self, multiplier, plot_raw_heatmap=False,save_raw_heatmap='raw_hits_heatmap.png', 
        plot_deconvolved_heatmap=False, save_deconvolve_heatmap='deconvolved_image.png', plot_signal_peak=False, 
        save_peak='peak.png',plot_conditions=False,check_resolved=False,condition='half_val'):
        
        # set multiplier
        self.multiplier = multiplier

        # shift origin
        self.shift_pos()

        # get heatmap
        self.get_raw()

        if plot_raw_heatmap:
            self.plot_heatmap(self.raw_heatmap, save_name=save_raw_heatmap)

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
            self.plot_heatmap(self.deconvolved_image, save_name=save_deconvolve_heatmap,label='signal')

        snr = np.amax(np.abs(self.deconvolved_image)) / np.std(np.abs(self.deconvolved_image))

        # get max signal (point sources usually)
        max_ind = np.where(self.deconvolved_image == np.amax(self.deconvolved_image))
        max_col = max_ind[1]
        if np.shape(max_col)[0] > 1:
            max_col = max_col[0]

        self.signal = np.fliplr(self.deconvolved_image)[:, int(max_col)]

        if plot_signal_peak:
            resolved = self.plot_peak(save_peak,plot_conditions,condition)
        elif check_resolved:
            _,resolved = self.check_resolved(condition)
        else:
            resolved = None

        return resolved




def shift(m, hs, vs):
    """
    m: input image
    hs: horizontal shift
    vs: vertical shift
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
