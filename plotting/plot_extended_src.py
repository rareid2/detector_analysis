import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cmocean

cmap = cmocean.cm.thermal


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


def polynomial_function(x, *coefficients):
    return np.polyval(coefficients, x)


# import FWHM
params = [2.52336124e-04, -2.83882554e-03, 8.86278977e-01]

# Create a 3x5 grid of subplots
fig, axs = plt.subplots(
    5, 3, figsize=(5.7, 8), gridspec_kw={"hspace": 0, "wspace": 0.12}
)


thetas = [2, 13, 24, 35, 46]
n_p = [5e6, 5e7, 5e8]
fcfov = 44.79

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

# now this is deviation from perfect

for i, n in enumerate(n_p):
    for j, theta in enumerate(thetas):
        n_particles = int((n * (5.265 * 2) ** 2) * (1 - np.cos(np.deg2rad(theta))))

        formatted_theta = "{:.0f}p{:02d}".format(int(theta), int((theta % 1) * 100))
        data = load_data(
            f"/home/rileyannereid/workspace/geant4/simulation-results/rings/59-3.47-{formatted_theta}-deg_{n_particles:.2E}_Mono_500_dc.txt"
        )

        # get in units of per cm^2 sr?
        # data = np.divide(data,gf_grid)
        # print(np.sum(data))
        geometric_factor = 18

        # get snr
        center_pixel = int(59 / 2)
        pixel_count = 59
        distance = 3.47
        pixel_size = 0.28 * 0.3  # cm
        total_signal = 0
        signal_count = 0

        for x in range(pixel_count):
            for y in range(pixel_count):
                relative_x = (x - center_pixel) * pixel_size
                relative_y = (y - center_pixel) * pixel_size

                aa = np.sqrt(relative_x**2 + relative_y**2)

                # find the geometrical theta angle of the pixel
                angle = np.arctan(aa / distance)
                angle = np.rad2deg(angle)

                if angle < (theta + 0.5):

                    total_signal += data[y, x]
                    # signal_count += 1
                    signal_count += gf_grid[y, x]

        # px_factor = 18*signal_count / (pixel_count**2)
        # print("recorded flux", total_signal)
        print(i, theta)
        print(total_signal / signal_count)
        # Plot heatmap
        im = axs[j, i].imshow(data / 18, cmap=cmap)

        # Add colorbar
        cbar = fig.colorbar(
            im,
            ax=axs[j, i],
            orientation="vertical",
            shrink=0.8,
            format=ticker.FuncFormatter(fmt),
            pad=0.02,
        )
        # ticks = np.linspace(np.min(data), np.max(data), 5)
        # cbar.set_ticks([ticks[1], ticks[3]])
        # cbar.set_ticklabels([ticks[1], ticks[3]])
        cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt))
        cbar.ax.tick_params(axis="y", labelsize=8)

        max_value = np.max(data / 18)
        power = np.log10(max_value)

        # Turn off axes by default
        axs[j, i].axis("off")

        # Turn on y-axes for subplots in the first column
        if i == 0:
            axs[j, i].get_yaxis().set_visible(True)

        # Turn on x-axes for subplots in the last row
        if j == 2:
            axs[j, i].get_xaxis().set_visible(True)

        axs[j, i].text(
            1.3,
            1.05,
            rf"$\times 10^{int(power)}$",
            ha="right",
            va="top",
            transform=axs[j, i].transAxes,
            fontsize=8,
        )


# Add subtitles to the left of each row
row_titles = [
    r"$\sim$8$\times$10$^7$cm$^{-2}$sr$^{-1}$",
    r"$\sim$8$\times$10$^6$cm$^{-2}$sr$^{-1}$",
    r"$\sim$8$\times$10$^5$cm$^{-2}$sr$^{-1}$",
]
for j in range(3):
    fig.text(
        0.23 + j * 0.265,
        0.885,
        row_titles[-1 * j - 1],
        va="center",
        ha="center",
        fontsize=10,
    )
col_titles = np.array(thetas) / fcfov
for i in range(5):
    fig.text(
        0.11,
        0.185 + i * 0.155,
        f"{round(100*np.flip(col_titles[-1*i-1]))}% FCFOV",
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=10,
    )
fig.text(0.29, 0.097, r"All colorbars have units of $cm^{-2}sr^{-1}$", fontsize=10)
plt.savefig(
    "../simulation-results/final-images/6_circles.png",
    dpi=500,
    bbox_inches="tight",
    pad_inches=0.02,
)
