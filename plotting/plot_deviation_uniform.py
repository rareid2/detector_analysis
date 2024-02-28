import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cmocean

cmap = cmocean.cm.thermal
colors = ["#39329E", "#88518D", "#D76C6B", "#FCA63B", "#E9F758"]
colors = ["#39329E", "#39329E", "#39329E", "#39329E", "#39329E"]

colors.reverse()


# Function to load 2D array from a txt file
def load_data(file_path):
    return np.loadtxt(file_path)


def fmt(x, pos):
    if x == 0:
        return "0"
    else:
        a, b = "{:.1e}".format(x).split("e")
        b = int(b)
        # print(b)
        a = float(a)
        if a % 1 > 0.1:
            pass
        else:
            a = int(a)
        return f"{a}"


# Create a 3x5 grid of subplots
fig, axs = plt.subplots(
    5,
    3,
    figsize=(5.7, 8),
    gridspec_kw={"hspace": 0, "wspace": 0},
    sharex=True,
    sharey=True,
)

thetas = [2, 13, 24, 35, 46]
n_p = [5e6, 1.5e7, 5e7]
fcfov = 44.79
distance = 3.47


def polynomial_function(x, *coefficients):
    return np.polyval(coefficients, x)


# import FWHM
params = [2.52336124e-04, -2.83882554e-03, 8.86278977e-01]
grid_size = 59
center = (grid_size - 1) / 2  # Center pixel in a 0-indexed grid

# Create a meshgrid representing the X and Y coordinates of each pixel
x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))

# Calculate the radial distance from the center for each pixel
radial_distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)

# now i have radiatl distance, use the FWHM thing
fwhm_grid = polynomial_function(radial_distance, *params)

# need to normalize to 1
fwhm_grid = 2 - (fwhm_grid / np.min(fwhm_grid))

# make it sum to 18
gf_grid = 18 * fwhm_grid / np.sum(fwhm_grid)

fwhm_step = 0
det_size_cm = 4.956  # cm
pixel_size = 0.28 * 0.3  # cm
max_rad_dist = np.sqrt(2) * det_size_cm / 2
bins = []
while pixel_size * fwhm_step < max_rad_dist:
    fwhm_z = polynomial_function(fwhm_step, *params)
    radial_distance_1 = fwhm_step * pixel_size
    angle1 = np.rad2deg(np.arctan(radial_distance_1 / distance))
    fwhm_step += fwhm_z

    radial_distance_2 = fwhm_step * pixel_size
    angle2 = np.rad2deg(np.arctan(radial_distance_2 / distance))
    # define bin edges using the step
    bin_edges = (angle1, angle2)
    bin_size = angle2 - angle1
    bins.append(angle2)
bins.insert(0, 0)
# bins = bins[::2]
bins = bins[:-1]
print(bins)
for i, n in enumerate(n_p):
    for j, theta in enumerate(thetas):
        fluxes = []
        calibrations = []
        stds = []
        uniformed = []
        n_particles = int((n * (5.265 * 2) ** 2) * (1 - np.cos(np.deg2rad(theta))))

        formatted_theta = "{:.0f}p{:02d}".format(int(theta), int((theta % 1) * 100))
        data = load_data(
            f"/home/rileyannereid/workspace/geant4/simulation-results/rings/59-3.47-{formatted_theta}-deg_{n_particles:.2E}_Mono_500_dc.txt"
        )
        # get snr
        center_pixel = int(59 / 2)
        pixel_count = 59
        snr_average = 0
        snr_num = 0
        rho = 0.5
        I = np.sum(data) * 0.993
        nt = 59**2
        ksi = ((0.007 * np.sum(data) + 100) / I) / nt

        slices = []
        slices_count = 0

        bins_ids = {f"{key}": [] for key in range(len(bins) - 1)}
        gf_ids = {f"{key}": [] for key in range(len(bins) - 1)}

        for x in range(pixel_count):
            for y in range(pixel_count):
                relative_x = (x - center_pixel) * pixel_size
                relative_y = (y - center_pixel) * pixel_size

                aa = np.sqrt(relative_x**2 + relative_y**2)

                # find the geometrical theta angle of the pixel
                angle = np.arctan(aa / distance)
                angle = np.rad2deg(angle)

                if angle < (theta + 0.5):
                    px_intensity = data[y, x]
                    psi = px_intensity / I
                    # if theta == 24:
                    #    print(psi)
                    snr = (
                        np.sqrt(nt * I)
                        * np.sqrt(rho * (1 - rho))
                        * psi
                        / (np.sqrt(rho + (1 - 2 * rho) * psi + ksi))
                    )
                    snr_average += snr
                    snr_num += 1
                # take a slice
                # lets do std of pitch angle bins - maybe make another figure
                for ii, bn in enumerate(bins[:-1]):
                    if angle >= bn and angle < bins[ii + 1]:
                        bins_ids[f"{ii}"].append(data[y, x])
                        gf_ids[f"{ii}"].append(gf_grid[y, x])
                    if ii == len(bins[:-1]):
                        if angle >= bn and angle <= bins[ii + 1]:
                            bins_ids[f"{ii}"].append(data[y, x])
                            gf_ids[f"{ii}"].append(gf_grid[y, x])
        # now get STD
        for ii, bn in enumerate(bins[:-1]):
            fluxes.append(
                np.average(np.array(bins_ids[f"{ii}"]))
                / np.average(np.array(gf_ids[f"{ii}"]))
            )
            # print(np.average(np.array(gf_ids[f"{ii}"])))
            stds.append(
                np.std(
                    np.array(bins_ids[f"{ii}"]) / np.average(np.array(gf_ids[f"{ii}"]))
                )
            )
            if bins[ii] < theta:
                uniformed.append(1)
            else:
                uniformed.append(0)

        # print(i, theta, snr_average / snr_num, I)

        normed_flux = np.array(fluxes) / max(fluxes)
        axs[j, i].plot(
            bins[:-1], uniformed, color="#D57965", linestyle="--", linewidth=2
        )
        axs[j, i].text(
            45.5,
            1.2,
            f"SNR: {round(snr_average / snr_num)}",
            ha="right",
            va="top",
            fontsize=8,
        )

        axs[j, i].plot(bins[:-1], normed_flux, color=colors[j], linewidth=2)
        y_upper = normed_flux + np.array(stds) / max(fluxes)
        y_lower = normed_flux - np.array(stds) / max(fluxes)

        axs[j, i].fill_between(
            bins[:-1],
            y_lower,
            y_upper,
            color=colors[j],
            alpha=0.3,
            label="Uncertainty 1",
        )
        axs[j, i].set_ylim([-0.25, 1.25])
        axs[j, i].tick_params(axis="y", labelsize=8)
        axs[j, i].tick_params(axis="x", labelsize=8)

# for i in range(5):
axs[2, 0].set_ylabel("Normalized Fluence", rotation="vertical", fontsize=10)
axs[2, 0].yaxis.labelpad = 0.2

# Add subtitles to the left of each row
row_titles = [
    r"$\sim$8$\times$10$^6$cm$^{-2}$sr$^{-1}$",
    r"$\sim$2$\times$10$^6$cm$^{-2}$sr$^{-1}$",
    r"$\sim$8$\times$10$^5$cm$^{-2}$sr$^{-1}$",
]
for j in range(3):
    fig.text(
        0.25 + j * 0.266,
        0.89,
        row_titles[-1 * j - 1],
        va="center",
        ha="center",
        fontsize=10,
    )
col_titles = np.array(thetas) / fcfov
for i in range(5):
    fig.text(
        0.92,
        0.188 + i * 0.155,
        f"{round(100*np.flip(col_titles[-1*i-1]))}% FCFOV",
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=10,
    )
fig.text(0.42, 0.072, r"Pitch Angle [$^\circ$]", fontsize=10)
plt.savefig(
    "../simulation-results/final-images/6p1_deviation.png",
    dpi=500,
    pad_inches=0.02,
    bbox_inches="tight",
)
