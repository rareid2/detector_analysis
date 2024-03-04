from simulation_engine import SimulationEngine
from hits import Hits
from scattering import Scattering
from deconvolution import Deconvolution
from scipy import interpolate
from macros import find_disp_pos
import numpy as np
import os
import subprocess

from scipy.io import savemat
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from plotting.plot_settings import *
from run_part_csv import read_csv_in_parts
from scipy.interpolate import interp1d

# construct = CA and TD
# source = DS and PS

simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=True)

txt = False
simulate = True
large_file = False

# general detector design
det_size_cm = 4.941  # cm
pixel = 0.81  # mm
pixel_size = pixel * 0.1

# ---------- coded aperture set up ---------
# set number of elements
n_elements_original = 61
multiplier = 1

element_size = pixel * multiplier
n_elements = (2 * n_elements_original) - 1

mask_size = element_size * n_elements
# no trim needed for custom design
trim = None
mosaic = True

# thickness of mask
thickness = 200  # um

# focal length
distance = 1.9764  # cm

# run each central theta

thetas = np.arange(3.327, 59.923, 3.327)
pixel_size = multiplier * pixel_size
geom_factor = 23.0955  # cm^2 sr
pitch_angle_data = [
    [4.0316625474712, 6310036.260811816],
    [7.570905419253705, 5123017.6427039],
    [10.949273615046103, 3899698.193624],
    [13.845017782868148, 2889552.074698],
    [16.419012598709976, 2198565.900710],
    [18.993007414551798, 1599654.825088],
    [21.406127554403504, 1200251.444176],
    [23.497498342274987, 915227.1626104],
    [26.07149315811681, 673502.43993565],
    [29.128112001928976, 487371.5094179],
    [32.02385616975103, 363059.27677150],
    [35.24134968955331, 276642.33868300],
    [38.780592561335816, 210456.4377596],
    [42.31983543311833, 171484.81923490],
    [45.859078304900834, 146627.4859799],
    [49.39832117668334, 129362.12920077],
    [52.937564048465845, 119331.6641433],
    [56.47680692024835, 110553.54005535],
    [60.016049792030856, 106044.0730941],
]


def polynomial_function(x, *coefficients):
    return np.polyval(coefficients, x)


# import the grid
params = [7.86180199e-05, 7.60095370e-03, 8.23512800e-01]
grid_size = 43
center = (grid_size - 1) / 2  # Center pixel in a 0-indexed grid
# Create a meshgrid representing the X and Y coordinates of each pixel
x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
# Calculate the radial distance from the center for each pixel
radial_distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)
# now i have radiatl distance, use the FWHM thing
fwhm_grid = polynomial_function(radial_distance, *params)
fwhm_grid = 2 - (fwhm_grid / np.min(fwhm_grid))
gf_grid = geom_factor * fwhm_grid / np.sum(fwhm_grid)

de_E = 0.17  # For Cassini CAPS ELS
central_energy = 235
x_data = [ld[0] for ld in pitch_angle_data]
y_data = [ld[1] * de_E for ld in pitch_angle_data]
interp_func = interp1d(x_data, y_data, kind="linear", fill_value="extrapolate")


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

fluxes = []
expected_fluxes = []
all_signal_raw = np.zeros((43, 43))

bins_ids = {f"{key}": [] for key in range(len(bins) - 1)}
gf_ids = {f"{key}": [] for key in range(len(bins) - 1)}

# ------------------- simulation parameters ------------------
for nn, theta in enumerate(thetas[:1]):
    pixel_count = int(n_elements_original)
    signal_count = 0
    center_pixel = int(pixel_count // 2)

    theta_bin = 3.327 / 2
    for x in range(pixel_count):
        for y in range(pixel_count):
            relative_x = (x - center_pixel) * pixel_size
            relative_y = (y - center_pixel) * pixel_size

            aa = np.sqrt(relative_x**2 + relative_y**2)

            # find the geometrical theta angle of the pixel
            angle = np.arctan(aa / distance)

            if (theta - theta_bin) < np.rad2deg(angle) < (theta + theta_bin):
                signal_count += gf_grid[y, x]

    px_factor = signal_count
    n_particles = int(10 * interp_func(theta) * geom_factor * px_factor)
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
    formatted_theta = "{:.0f}p{:02d}".format(int(theta), int((theta % 1) * 100))
    fname_tag = f"{n_elements_original}-{distance}-{formatted_theta}-deg"

    fname = f"../simulation-data/strahl/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_{formatted_theta}.csv"

    if txt:
        fname = f"../simulation-results/strahl/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_raw.txt"

    fname = "../simulation-results/strahl/all_raw.txt"

    simulation_engine.set_macro(
        n_particles=int(n_particles),
        energy_keV=[energy_type, energy_level, None],
        surface=True,
        progress_mod=int(n_particles / 10),  # set with 10 steps
        fname_tag=fname_tag,
        confine=False,
        detector_dim=det_size_cm,
        theta=theta,
        ring=True,
        # radius_cm=3,
    )

    # --------------RUN---------------
    if simulate:
        simulation_engine.run_simulation(fname, build=False, rename=True)

    # ---------- process results -----------
    # directory to save results in
    results_dir = "../simulation-results/strahl/"
    results_tag = f"{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}"
    results_tag = "all_thetas"
    results_save = results_dir + results_tag

    if large_file and not txt:
        raw_hits = read_csv_in_parts(
            fname,
            fname_tag,
            simulation_engine,
            n_elements_original,
            energy_level,
            multiplier,
        )
        np.savetxt(results_save + "_raw.txt", raw_hits)
        txt = True
        fname = results_save + "_raw.txt"

    myhits = Hits(fname=fname, experiment=False, txt_file=txt)
    if not txt:
        _, sec_brehm, sec_e = myhits.get_det_hits(
            remove_secondaries=False, second_axis="y", energy_level=energy_level
        )
        # print(sec_brehm, sec_e, len(myhits.hits_dict["Position"]))

    # deconvolution steps
    deconvolver = Deconvolution(myhits, simulation_engine)

    deconvolver.deconvolve(
        downsample=int(n_elements_original),
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
    print(np.sum(deconvolver.deconvolved_image))
    signal = deconvolver.deconvolved_image
    total_count = 0

    for x in range(pixel_count):
        for y in range(pixel_count):
            relative_x = (x - center_pixel) * pixel_size
            relative_y = (y - center_pixel) * pixel_size

            aa = np.sqrt(relative_x**2 + relative_y**2)

            # find the geometrical theta angle of the pixel
            angle = np.arctan(aa / distance)
            angle = np.rad2deg(angle)

            """
            if (theta - theta_bin) < angle < (theta + theta_bin):
                total_count += signal[y, x]

                rect = patches.Rectangle(
                    (x - 0.5, y - 0.5),
                    1,
                    1,
                    linewidth=2,
                    edgecolor="black",
                    facecolor="none",
                )
            """

            for ii, bn in enumerate(bins[:-1]):
                if angle >= bn and angle < bins[ii + 1]:
                    bins_ids[f"{ii}"].append(signal[y, x])
                    gf_ids[f"{ii}"].append(gf_grid[y, x])
                if ii == len(bins[:-1]):
                    if angle >= bn and angle <= bins[ii + 1]:
                        bins_ids[f"{ii}"].append(data[y, x])
                        gf_ids[f"{ii}"].append(gf_grid[y, x])

                # plt.gca().add_patch(rect)
    # print("recorded flux", total_count / (geom_factor * px_factor))
    # print("expected flux", interp_func(theta))
    # fluxes.append(total_count / (geom_factor * px_factor))
    expected_fluxes.append(interp_func(theta))

    # plt.imshow(signal, cmap=cmap)
    # plt.colorbar()
    # plt.show()
    # all_signal_raw += np.loadtxt(results_save + "_raw.txt")

theta_centers = []
for ii, bn in enumerate(bins[:-1]):
    fluxes.append(
        np.average(np.array(bins_ids[f"{ii}"])) / np.average(np.array(gf_ids[f"{ii}"]))
    )
    theta_centers.append((bins[ii + 1] + bn) / 2)

plt.scatter(theta_centers, np.log10(fluxes))
# plt.scatter(theta_centers, np.log10(expected_fluxes))
plt.savefig("../simulation-results/strahl/flux_bins.png")

# np.savetxt("../simulation-results/strahl/all_raw.txt", all_signal_raw)
