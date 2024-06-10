import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap
import matplotlib.ticker as mtick
from numpy.typing import NDArray
import numpy as np
import sys
import matplotlib.patches as patches
from scipy import signal

sys.path.insert(1, "../coded_aperture_mask_designs")
from util_fncs import makeMURA, make_mosaic_MURA, get_decoder_MURA, updated_get_decoder


def resample(array):
    # resample the array into desired size

    original_size = len(array)

    multiplier = 2

    new_array = np.zeros((len(array) // multiplier, len(array) // multiplier))
    for i in range(0, original_size, multiplier):
        k = i // multiplier
        for j in range(0, original_size, multiplier):
            n = j // multiplier
            new_array[k, n] = np.sum(array[i : i + multiplier, j : j + multiplier])

    return new_array


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


def fft_conv(im1, im2):

    # Fourier space multiplication
    Image = np.real(np.fft.ifft2(np.fft.fft2(im1) * np.fft.fft2(im2)))

    # Shift to by half of image length after convolution
    deconvolved_image = shift(Image, len(im2) // 2, len(im2) // 2)

    return deconvolved_image


results_folder = "/home/rileyannereid/workspace/geant4/experiment_results/"
import cmocean

# need to get the mask
mask, _ = make_mosaic_MURA(
    11,
    1.21,
    holes=False,
    generate_files=False,
)
mask = mask[5:16, 5:16]
# mask = np.repeat(
#    np.repeat(mask, 11, axis=1),
#    11,
#    axis=0,
# )

rawim = np.loadtxt(f"{results_folder}Cd109-0_raw.txt")
# rawim = np.loadtxt(
#    "/home/rileyannereid/workspace/geant4/experiment_results/Cd109-0-7_cleaned.txt"
# )
rawim = resample(rawim[8:250, 8:250])
# rawim = resample(rawim)
# for i in range(len(rawim)):
#    for j in range(len(rawim[i])):
#        if rawim[i][j] > 15:
#            rawim[i][j] = 1
print(np.sum(rawim))
plt.clf()
plt.imshow(rawim)
plt.colorbar()
plt.savefig("text.png")

flipped_decoder = np.fliplr(np.flipud(mask))
unflipped_decoder = mask

mask = np.fliplr(mask)

iguess = np.ones_like(rawim) * 0.5
guess = iguess
for i in range(30):
    forward = signal.convolve2d(guess, mask, "same")
    relative_diff = rawim / (forward + (np.ones_like(forward) * 1e-7))
    back = signal.correlate2d(relative_diff, mask, "same")
    guess = np.multiply(guess, back)
print(np.sum(guess))
cmap = cmocean.cm.thermal
inds = ["0_raw", "0-background_dc"]

cmap = cmocean.cm.thermal
# Create subplots with shared x-axis and y-axis
fig, axs = plt.subplots(1, 3, figsize=(8, 3), sharex="col")

panels = ["i", "b)", "c)"]
im = axs[2].imshow(guess, cmap=cmap)
axs[2].xaxis.set_visible(False)
axs[2].yaxis.set_visible(False)
label = "Reconstructed Counts"
extend = "neither"
fraction = 0.047
cbar = fig.colorbar(
    im,
    ax=axs[2],
    orientation="vertical",
    fraction=0.047,
    pad=0.01,
    label=label,
    extend=extend,
)
# Plot data and colorbars
for i in range(2):
    fname = f"{results_folder}Cd109-{inds[i]}.txt"
    print(fname)
    if i > 0:
        data = np.loadtxt(fname) / 11**2
        print(np.sum(data))
    else:
        data = np.loadtxt(fname)
    ax = axs[i]
    ax.set_box_aspect(1)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    if i == 0:
        vmax = 100
        label = "keV/pixel"
        extend = "max"
        fraction = 0.046
    else:
        vmax = None
        label = "Reconstructed Counts"
        extend = "neither"
        fraction = 0.047

    if i < 2:  # First two columns
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmax=vmax)
        cbar = fig.colorbar(
            im,
            ax=ax,
            orientation="vertical",
            fraction=fraction,
            pad=0.01,
            label=label,
            extend=extend,
        )
        cbar.ax.yaxis.labelpad = 1.2

    else:  # Last column
        im = ax.imshow(data, aspect="auto", cmap=cmap)
        cbar = fig.colorbar(
            im,
            ax=ax,
            orientation="vertical",
            fraction=0.05,
            pad=0.01,
            label=label,
        )
        cbar.ax.yaxis.labelpad = 1.2
    if i < 1:
        ax.text(6, 20, "a)", c="white")
    else:
        ax.text(3, 10, panels[i], c="white")


# Adjust layout
fig.tight_layout()
plt.savefig(
    f"{results_folder}decoding.png",
    dpi=500,
    bbox_inches="tight",
)

# print(np.sum(np.loadtxt(f"{results_folder}Cd109-0_cleaned.txt")))
# print(np.sum(np.loadtxt(f"{results_folder}Cd109-0-background_raw.txt")))

# we should calculate SNR
plt.clf()
signal = np.loadtxt(f"{results_folder}Cd109-0-background_dc.txt") / 11**2
signal = guess
pixel_count = int(11 * 11)

signal_count = 0
total_count = 0
center_pixelx = int(11 * 11 / 2) + 3
center_pixely = int(11 * 11 / 2) - 2

pixel_size = 0.0605

nt = 121**2
ksi = 0.0102 * 60 * 104 / (11 * 11) ** 2  # background counts after cleaning
ksi = (8029 - 2144) / (11 * 11) ** 2

distance = 2
rho = 0.5
theta = 11
snr_average = 0
snr_num = 0

I = np.sum(np.loadtxt(f"{results_folder}Cd109-0-background_cleaned.txt")) - (
    ksi * (11 * 11) ** 2
)
print("source counts", I)

for x in range(pixel_count):
    for y in range(pixel_count):
        relative_x = (x - center_pixelx) * pixel_size
        relative_y = (y - center_pixely) * pixel_size

        aa = np.sqrt(relative_x**2 + relative_y**2)

        # find the geometrical theta angle of the pixel
        angle = np.arctan(aa / distance)

        if np.rad2deg(angle) < (theta + 2):
            px_intensity = signal[y, x]
            print(signal[y, x])
            psi = px_intensity / I

            rect = patches.Rectangle(
                (x - 0.5, y - 0.5),
                1,
                1,
                linewidth=2,
                edgecolor="black",
                facecolor="none",
            )
            plt.gca().add_patch(rect)

            snr = np.sqrt(nt * I) * 0.5 * psi / (np.sqrt(0.5 + ksi))
            snr_average += snr
            snr_num += 1
print("snr", snr_average / snr_num)
# plt.imshow(signal)
# plt.show()
