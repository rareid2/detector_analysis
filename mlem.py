import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

# load raw image
int_time = 10
fname_tag = f"all_hits_broad_{int_time}"
fname = f"../simulation-results/strahl/{fname_tag}_raw.txt"
fname_dc = f"../simulation-results/strahl/{fname_tag}_dc.txt"

decoder_fname = "../simulation-results/strahl/decoder.txt"


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


def resample(array):
    # resample the array into desired size

    original_size = len(array)

    multiplier = original_size // 59

    new_array = np.zeros((len(array) // multiplier, len(array) // multiplier))

    for i in range(0, original_size, multiplier):
        k = i // multiplier
        for j in range(0, original_size, multiplier):
            n = j // multiplier
            new_array[k, n] = np.sum(array[i : i + multiplier, j : j + multiplier])

    return new_array


"""
rawim = np.loadtxt(fname)
dcim_true = np.loadtxt(fname_dc)
# dcim_true = resample(dcim_true)
dcim = np.ones_like(dcim_true) * 1
# dcim = np.random.rand(*(59, 59)) - 0.5

decoder = np.loadtxt(decoder_fname)
# decoder[decoder == 0] = -1

# rawim = resample(rawim)

cc_grid = np.zeros_like(rawim)
pixel_count = 59 * 3
pixel_size = 0.028  # cm
distance = 0.923  # cm
center_pixel = 59 * 3 // 2
thickness = 0.01  # cm

for x in range(pixel_count):
    for y in range(pixel_count):
        relative_x = (x - center_pixel) * pixel_size
        relative_y = (y - center_pixel) * pixel_size

        aa = np.sqrt(relative_x**2 + relative_y**2)

        # find the geometrical theta angle of the pixel
        angle = np.arctan(aa / distance)

        # calculate d
        d = np.tan(angle) * thickness

        # compare
        d = min([d, 2 * pixel_size])

        # find factor
        cc = d / pixel_size

        cc_grid[y, x] = 1 - cc

rawim = rawim  # / np.rot90(cc_grid)
print(np.shape(decoder))
plt.imshow(decoder)
plt.savefig("decoder.png")
flipped_decoder = np.fliplr(np.rot90(decoder))
unflipped_decoder = decoder

iguess = dcim
guess = iguess
for i in range(800):
    forward = fft_conv(guess, unflipped_decoder)

    relative_diff = rawim / (forward + (np.ones_like(forward) * 1e-7))

    back = fft_conv(relative_diff, flipped_decoder)

    guess = np.multiply(guess, back)

    if i == 10:
        guess_10 = guess
    if i == 30:
        guess_50 = guess

import cmocean

cmap = cmocean.cm.thermal

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(7, 4))
im = ax1.imshow(dcim_true, cmap=cmap)
cbar = fig.colorbar(im, ax=ax1, orientation="horizontal", fraction=0.04, pad=0.01)
cbar.ax.tick_params(axis="x", labelsize=8)
im = ax2.imshow(iguess, cmap=cmap)
cbar = fig.colorbar(im, ax=ax2, orientation="horizontal", fraction=0.04, pad=0.01)
cbar.ax.tick_params(axis="x", labelsize=8)
im = ax3.imshow(guess_10, cmap=cmap)
cbar = fig.colorbar(im, ax=ax3, orientation="horizontal", fraction=0.04, pad=0.01)
cbar.ax.tick_params(axis="x", labelsize=8)
im = ax4.imshow(guess_50, cmap=cmap)
cbar = fig.colorbar(im, ax=ax4, orientation="horizontal", fraction=0.04, pad=0.01)
cbar.ax.tick_params(axis="x", labelsize=8)
im = ax5.imshow(guess, cmap=cmap)
cbar = fig.colorbar(im, ax=ax5, orientation="horizontal", fraction=0.04, pad=0.01)
cbar.ax.tick_params(axis="x", labelsize=8)
ax1.axis("off")
ax2.axis("off")
ax3.axis("off")
ax4.axis("off")
ax5.axis("off")

ax1.text(2, 7, "MURA Decoding", fontsize=8, color="white")
ax2.text(2, 7, r"MLEM $f^0$", fontsize=8, color="white")
ax3.text(2, 7, r"MLEM $f^{10}$", fontsize=8, color="white")
ax4.text(2, 7, r"MLEM $f^{100}$", fontsize=8, color="white")
ax5.text(2, 7, r"MLEM $f^{200}$", fontsize=8, color="white")
print(np.sum(guess_50))
print(np.sum(dcim_true) / 9)

np.savetxt("../simulation-results/strahl/mlem_200_dc.txt", guess)
plt.subplots_adjust(wspace=0.02, hspace=0)
plt.savefig(
    "../simulation-results/strahl/mlem.png",
    dpi=500,
    bbox_inches="tight",
    pad_inches=0.02,
)
"""
