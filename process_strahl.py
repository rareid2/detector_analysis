from simulation_engine import SimulationEngine
from hits import Hits
from scattering import Scattering
from deconvolution import Deconvolution
from scipy import interpolate
from macros import find_disp_pos
import numpy as np
import os
import subprocess
import copy
from scipy.io import savemat
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from plotting.plot_settings import *
from run_part_csv import read_csv_in_parts
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from matplotlib.patches import FancyArrowPatch
import cmocean

cmap = cmocean.cm.thermal

data = [
    [1.7469814290210905, 6843.7],
    [5.8801586402266395, 6024.8],
    [9.59614730878188, 4377.040],
    [13.079886685552417, 3109.1],
    [16.176543909348446, 2241.2],
    [19.273201133144482, 1543.5],
    [22.17631728045326, 1086.47],
    [25.07943342776204, 743.318],
    [28.176090651558077, 540.44],
    [31.466288951841364, 385.90],
    [35.33711048158641, 275.110],
    [39.59501416430595, 202.321],
    [43.852917847025495, 161.08],
    [48.11082152974504, 136.256],
    [52.36872521246458, 122.458],
    [56.626628895184126, 114.59],
    [60.884532577903684, 108.01],
    [65.14243626062323, 110.058],
    [69.40033994334277, 104.789],
    [73.65824362606232, 105.092],
    [77.91614730878186, 102.103],
    [82.1740509915014, 102.6941],
    [86.43195467422095, 103.736],
    [90.6898583569405, 104.7895],
    [94.94776203966005, 104.638],
    [99.20566572237959, 104.336],
    [103.46356940509914, 104.33],
    [107.72147308781868, 105.09],
    [112.72, 104.01453145693937],
    [120.10810198300283, 103.91],
    [125.72079320113312, 102.10],
    [131.33348441926344, 102.10],
    [136.7526345609065, 105.157],
    [142.51048158640225, 103.73],
    [147.76, 104.01453145693937],
    [153.35999999999999, 101.32],
    [158.64, 101.32065897066866],
    [163.8483852691218, 102.103],
    [169.26753541076485, 102.10],
    [173.71898016997164, 102.10],
]

all_fd = []
int_time = 10
sigma_fits = np.arange(9, 30, 2)
# for sigma_fit in sigma_fits:
xd = np.array([d[0] for d in data])
yd = [d[1] * 10**3 * 0.17 * int_time for d in data]


def gaussian(x, sigma, A):
    halo = 104 * 10**3 * 0.17 * int_time
    return halo + A * np.exp(-((x - 0) ** 2) / (2 * sigma**2))


# Fit the Gaussian function to the data
p0 = [11, 1e6 * int_time]  # Initial guess for the parameters [mu, sigma, A]
params, cov_matrix = curve_fit(gaussian, xd, yd, p0=p0)
# print(params)
interp_func = interp1d(xd, yd, kind="linear", fill_value="extrapolate")
sigma_fit, A_fit = params
sigma_fit = 30
simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=True)

txt = True
simulate = False

# general detector design
det_size_cm = 4.94  # cm
pixel = 3.8  # mm
pixel_size = pixel * 0.1

# ---------- coded aperture set up ---------
# set number of elements
n_elements_original = 13
multiplier = 1

element_size = pixel * multiplier
n_elements = (2 * n_elements_original) - 1

mask_size = element_size * n_elements
# no trim needed for custom design
trim = None
mosaic = True

# thickness of mask
thickness = 100  # um

# focal length
distance = 4.3  # cm

# run each central theta
pixel_size = multiplier * pixel_size
geom_factor = 18.9276  # cm^2 sr


def polynomial_function(x, *coefficients):
    return np.polyval(coefficients, x)


# import the grid
params = [1.48515239e-04, 6.90018731e-03, 8.08592346e-01]
params = [1.26999169e-03, -1.95243018e-02, 3.28582722e00]

grid_size = 13 * 3
center = (grid_size - 1) / 2  # Center pixel in a 0-indexed grid
# Create a meshgrid representing the X and Y coordinates of each pixel
x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
# Calculate the radial distance from the center for each pixel
radial_distance = np.sqrt(((x - center) / 3) ** 2 + ((y - center) / 3) ** 2)
# now i have radiatl distance, use the FWHM thing
fwhm_grid = polynomial_function(radial_distance, *params)
fwhm_grid = 2 - (fwhm_grid / np.amin(fwhm_grid))

gf_grid = geom_factor * fwhm_grid / np.sum(fwhm_grid)

# make bins
fwhm_step = 0
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
bins = bins[:-1]

bins = np.arange(0, 37, 5.05)  # bin edges
bins_ids = {f"{key}": [] for key in range(len(bins) - 1)}
gf_ids = {f"{key}": [] for key in range(len(bins) - 1)}

# ------------------- simulation parameters ------------------
pixel_size = 0.12666666666666666667
pixel_count = int(n_elements_original * 3)
center_pixel = int(pixel_count // 2)

# --------------set up simulation---------------
simulation_engine.set_config(
    det1_thickness_um=300,
    det_gap_mm=30,  # gap between first and second (unused detector)
    win_thickness_um=100,  # window is not actually in there
    det_size_cm=det_size_cm,
    n_elements=n_elements,
    mask_thickness_um=thickness,
    mask_gap_cm=distance,
    element_size_mm=element_size,
    mosaic=mosaic,
    mask_size=mask_size,
    radius_cm=1,
)

# --------------set up source---------------
energy_type = "Mono"
energy_level = 0.235  # keV

# --------------set up data naming---------------
n_particles = 0
fname_tag = f"all_hits_13_{int_time}_{sigma_fit}"

simulation_engine.set_macro(
    n_particles=int(n_particles),
    energy_keV=[energy_type, energy_level, None],
    surface=True,
    progress_mod=int(n_particles / 10),  # set with 10 steps
    fname_tag=fname_tag,
    confine=False,
    detector_dim=det_size_cm,
    theta=0,
    theta_lower=0,
    ring=True,
    plane_size_cm=6.234,
    # radius_cm=3,
)

# ---------- process results -----------
# directory to save results in
results_dir = "../simulation-results/strahl/"
results_tag = f"{fname_tag}"
results_save = results_dir + results_tag

myhits = Hits(fname=results_save + "_raw.txt", experiment=False, txt_file=txt)

# deconvolution steps
deconvolver = Deconvolution(myhits, simulation_engine)

deconvolver.deconvolve(
    downsample=int(n_elements_original * 3),
    trim=trim,
    vmax=None,
    plot_deconvolved_heatmap=True,
    plot_raw_heatmap=True,
    save_raw_heatmap=results_save + "_raw.png",
    save_deconvolve_heatmap=results_save + "_dc.png",
    plot_signal_peak=False,
    plot_conditions=False,
    flat_field_array=None,
    hits_txt=txt,
    rotate=True,
    delta_decoding=False,
    apply_noise=False,
    resample_array=True,
)
np.savetxt(results_save + "_dc.txt", deconvolver.deconvolved_image)
print(np.sum(deconvolver.raw_heatmap))
print(np.sum(deconvolver.deconvolved_image) / 9)
negative_angles = []

signal = deconvolver.deconvolved_image / 9
# signal = np.loadtxt("../simulation-results/strahl/mlem_200_dc.txt")
for x in range(pixel_count):
    for y in range(pixel_count):
        relative_x = (x - center_pixel) * pixel_size
        relative_y = (y - center_pixel) * pixel_size

        aa = np.sqrt(relative_x**2 + relative_y**2)

        # find the geometrical theta angle of the pixel
        angle = np.arctan(aa / distance)
        angle = np.rad2deg(angle)

        for ii, bn in enumerate(bins[:-1]):
            if angle >= bn and angle < bins[ii + 1]:
                bins_ids[f"{ii}"].append(signal[y, x])
                gf_ids[f"{ii}"].append(gf_grid[y, x])
            if ii == len(bins[:-1]):
                if angle >= bn and angle <= bins[ii + 1]:
                    bins_ids[f"{ii}"].append(signal[y, x])
                    gf_ids[f"{ii}"].append(gf_grid[y, x])

        if signal[y, x] < 0:
            negative_angles.append(angle)
print("minimum negative angle", min(negative_angles))
# now plot results
fluxes = []
bin_plot = []
central_bins = []
fd = []
for ii, bn in enumerate(bins[:-1]):
    flux = np.sum(np.array(bins_ids[f"{ii}"])) / np.sum(np.array(gf_ids[f"{ii}"]))
    flux_diff = gaussian((bins[ii + 1] + bn) / 2, sigma_fit, A_fit) - flux
    fd.append(100 * (flux_diff / gaussian((bins[ii + 1] + bn) / 2, sigma_fit, A_fit)))
    central_bins.append((bins[ii + 1] + bn) / 2)
    bin_plot.append(bn)
    bin_plot.append(bins[ii + 1])
    fluxes.append(flux)
    fluxes.append(flux)

central_bins = np.array(central_bins)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.8))
im = ax1.imshow(signal, cmap=cmap)
ax1.axis("off")
cbar = fig.colorbar(im, ax=ax1, orientation="horizontal", fraction=0.047, pad=0.01)
cbar.set_label(r"Fluence [cm$^{-2}$ sr$^{-1}$]", fontsize=8)
cbar.ax.xaxis.labelpad = 1.2
cbar.ax.tick_params(axis="x", labelsize=8)
ax2.set_ylim([5e5, 0.15e8])
ax2.plot(central_bins, gaussian(central_bins, sigma_fit, A_fit), color="#D57965")
ax2.plot(bin_plot, fluxes, color="#39329E")
ax2.set_ylabel(r"Fluence [cm$^{-2}$ sr$^{-1}$]", fontsize=8)
ax2.set_xlabel("Pitch Angle [deg]", fontsize=8)
ax2.set_yscale("log")
ax2.xaxis.labelpad = 1.2
ax2.yaxis.labelpad = 1.2
# arrow = FancyArrowPatch((0.3, 0.8), (0.7, 0.8), arrowstyle='<->', color='#D57965', lw=1, mutation_scale=20, transform=ax2.transAxes)
# ax2.add_patch(arrow)
# ax2.text(40,5e6,r'w=11$^{\circ}$',color='#D57965',fontsize=8)
ax2.tick_params(axis="both", labelsize=8)
# ax2.set_ylim([1e0, 2e6])

plt.savefig(
    f"../simulation-results/strahl/{results_tag}_distribution.png",
    dpi=500,
    bbox_inches="tight",
    pad_inches=0.02,
)
# all_fd.append(fd)

"""
plt.clf()
for fi, fd in enumerate(all_fd):
    plt.plot(central_bins, fd, label=f"{int(sigma_fits[fi])}")

plt.xlabel("Pitch Angle [deg]")
plt.ylabel("percent diff in \n reconstructed vs simulated flux")

plt.legend()
plt.savefig(
    f"../simulation-results/strahl/all_percent_diff.png",
    dpi=500,
    bbox_inches="tight",
    pad_inches=0.02,
)
"""
