import numpy as np
import cmocean

cmap = cmocean.cm.thermal

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14})


def fmt(x, pos):
    if x == 0:
        return "0"
    else:
        a, b = "{:.1e}".format(x).split("e")
        b = int(b)
        a = float(a)
        if a % 1 > 0.1:
            pass
        else:
            a = int(a)
        return f"{a}"


def fcfov_plane(
    detector_size_mm: float,
    mask_size_mm: float,
    mask_detector_distance_mm: float,
    mask_plane_distance_mm: float,
):
    detector_diagonal_mm = detector_size_mm * np.sqrt(2)
    mask_diagonal_mm = mask_size_mm * np.sqrt(2)

    detector_half_diagonal_mm = detector_diagonal_mm / 2
    mask_half_diagonal_mm = mask_diagonal_mm / 2

    # FCFOV half angle
    theta_fcfov_deg = np.rad2deg(
        np.arctan(
            (mask_half_diagonal_mm - detector_half_diagonal_mm)
            / mask_detector_distance_mm
        )
    )
    # print(theta_fcfov_deg, "half angle")

    # pcfov
    fov = np.rad2deg(
        np.arctan(
            (detector_diagonal_mm + (mask_half_diagonal_mm - detector_half_diagonal_mm))
            / mask_detector_distance_mm
        )
    )
    # print("PCFOV", fov - theta_fcfov_deg, "Half angle pcfov")
    # project this to a distance
    plane_distance_to_detector_mm = mask_detector_distance_mm + mask_plane_distance_mm

    additional_diagonal_mm = (
        np.tan(np.deg2rad(theta_fcfov_deg)) * plane_distance_to_detector_mm
    )

    plane_diagonal_mm = (additional_diagonal_mm + detector_half_diagonal_mm) * 2

    plane_side_length_mm = plane_diagonal_mm / np.sqrt(2)

    # geant asks for half side length

    plane_half_side_length_mm = plane_side_length_mm / 2

    # print(f"FCFOV square plane should be half side length = {plane_half_side_length_mm} mm at a distance {plane_distance_to_detector_mm} mm from detector")

    pcfov = fov - theta_fcfov_deg
    return pcfov, theta_fcfov_deg


n_particles_frac = np.logspace(0, -2, 8)
npf = n_particles_frac[1]  # used 1 in the paper
print(npf / n_particles_frac[1])
n_particles = (
    int((8e7 * (5.030 * 2) ** 2) * (np.cos(np.deg2rad(45)) - np.cos(np.deg2rad(70))))
    * npf
)
detector_size_mm = 49.56
mask_size_mm = 98.28
mask_detector_distance_mm = 34.7
element_size_mm = 0.84
mask_plane_distance_mm = 0.15 + 1.0074 + 0.5

mask_detector_distance_mms = np.linspace(1, 100, 50)

fcfovs = []
pcfovs = []
fovs = []

for mask_detector_distance_mm in mask_detector_distance_mms:
    pcfov, fcfov = fcfov_plane(
        detector_size_mm,
        mask_size_mm,
        mask_detector_distance_mm,
        mask_plane_distance_mm,
    )
    fcfovs.append(fcfov)
    pcfovs.append(pcfov)
    fovs.append(pcfov / (fcfov / 2))
fcfov_plane(detector_size_mm, mask_size_mm, 34.7, mask_plane_distance_mm)

# load data
pcfov_rotated = np.loadtxt(
    f"../simulation-results/rings/59-3.47-45p00-deg-rotate_{n_particles:.2E}_Mono_100_dc.txt"
)
pcfov = np.loadtxt(
    f"../simulation-results/rings/59-3.47-45p00-deg_{n_particles:.2E}_Mono_100_dc.txt"
)
signal = np.loadtxt(
    "../simulation-results/rings/59-3.47-22p00-deg_5.90E+08_Mono_100_dc.txt"
)
# signal = np.loadtxt("../simulation-results/rings/59-3.47-35p00-deg_1.00E+09_Mono_100_dc.txt")
cpfov = pcfov + pcfov_rotated

combined = np.loadtxt(
    "../simulation-results/rings/59-3.47-22-45-deg-fov_1.53E+09_Mono_100_combined.txt"
)

pixel_count = 59
center_pixel = int(59 / 2)
geometric_factor = 18
pcfov_signal = 0
fcfov_signal = 0
signal_count = 0
geometric_factor = 18
distance = 3.47
pixel = 0.28 * 3  # mm
pixel_size = pixel * 0.1
for x in range(pixel_count):
    for y in range(pixel_count):
        relative_x = (x - center_pixel) * pixel_size
        relative_y = (y - center_pixel) * pixel_size

        aa = np.sqrt(relative_x**2 + relative_y**2)

        # find the geometrical theta angle of the pixel
        angle = np.arctan(aa / distance)

        if (22 - 1.4) < np.rad2deg(angle) < (22 + 0.5):
            signal_count += 1
            fcfov_signal += signal[y, x]
            pcfov_signal += signal[y, x] + pcfov[y, x]

px_factor = signal_count / (pixel_count**2)

print("recorded flux", fcfov_signal / (geometric_factor * px_factor))
print("new flux", pcfov_signal / (geometric_factor * px_factor))
print(
    "error",
    100
    - (
        100
        * (
            (pcfov_signal / (geometric_factor * px_factor))
            / (fcfov_signal / (geometric_factor * px_factor))
        )
    ),
)

cpfov = pcfov + pcfov_rotated
datum = [signal, pcfov, pcfov_rotated, cpfov, combined]

fig, axs = plt.subplots(5, 2, figsize=(5, 8))  # 1 row, 2 columns
letters = ["a", "c", "e", "g", "i"]
for i in range(5):
    data = datum[i]
    # Plot heatmap
    im = axs[i, 0].imshow(data / 18, cmap=cmap)

    # Add colorbar
    cbar = fig.colorbar(
        im,
        ax=axs[i, 0],
        orientation="vertical",
        shrink=0.8,
        format=ticker.FuncFormatter(fmt),
        pad=0.02,
    )
    cbar.set_label(label=r"cm$^{-2}$sr$^{-1}$", size=8)

    # ticks = np.linspace(np.min(data), np.max(data), 5)
    # cbar.set_ticks([ticks[1], ticks[3]])
    # cbar.set_ticklabels([ticks[1], ticks[3]])
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt))
    cbar.ax.tick_params(axis="y", labelsize=8)
    cbar.ax.yaxis.labelpad = 2

    max_value = np.max(data / 18)
    power = np.log10(max_value)

    # Turn off axes by default
    axs[i, 0].axis("off")
    axs[i, 1].axis("off")

    axs[i, 0].text(
        1.3,
        1.02,
        rf"$\times 10^{int(power)}$",
        ha="right",
        va="top",
        transform=axs[i, 0].transAxes,
        fontsize=8,
    )
    axs[i, 0].text(
        0.13,
        0.95,
        f"{letters[i]})",
        ha="right",
        va="top",
        transform=axs[i, 0].transAxes,
        color="white",
        fontsize=8,
    )


# axes[0,0].text(1.3, 1.01, rf'$\times 10^{int(3)}$', ha='right', va='top', transform=axes[0].transAxes, fontsize=8)
# axes[0,0].text(0.97, -0.03,  r"$50\%$ FCFOV + PCFOV" + "\n" + "Reconstructed image", ha='right', va='top', transform=axes[0].transAxes, fontsize=10)
# axes[0,0].text(0.08, 0.98, "a", ha='right', va='top', transform=axes[0].transAxes, fontsize=10)

# Subplot 2
# axs[-1,:].plot(mask_detector_distance_mms / detector_size_mm, fovs, "#9b2226")
"""
axes[1,0].plot(mask_detector_distance_mms / detector_size_mm, fovs, "#9b2226")
#axes[1].set_aspect('equal', adjustable='datalim')
cbar.ax.yaxis.labelpad = 1.2

axes[1,0].xaxis.labelpad = 0.2
axes[1,0].yaxis.labelpad = 0.2

axes[1,0].set_xlim([0,2])
axes[1,0].set_ylim([0,3])
axes[1,0].grid(True, linestyle='--', alpha=0.6)  # Add grid lines with transparency
axes[1,0].set_xlabel("f-number",fontsize=10)
axes[1,0].set_ylabel("PCFOV / Half FCFOV",fontsize=10)
fig.subplots_adjust(wspace=0.35)  # Adjust the horizontal space between subplots
#axes[1].set_position([0.6, 0.2, 0.3, 0.6]) 
axes[1].tick_params(axis='both', which='both', labelsize=8)
axes[1].text(0.08, 0.98, "b", ha='right', va='top', transform=axes[1].transAxes, fontsize=10)
"""
plt.savefig(
    "../simulation-results/final-images/9_fcfov_ratio.png",
    dpi=500,
    bbox_inches="tight",
    pad_inches=0.02,
)
