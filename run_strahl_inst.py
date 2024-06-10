# run the WHOLE design!
from simulation_engine import SimulationEngine
from mlem import fft_conv
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
import sys
from run_flatfield import run_geom_corr
import scipy
import cmocean


sys.path.insert(1, "../coded_aperture_mask_designs")
from create_mask import generate_CA
from util_fncs import makeMURA, make_mosaic_MURA, get_decoder_MURA, updated_get_decoder


def apply_ps_noise(raw_img, nn):
    detector_noise = dark_current(raw_img, nn / ((3 * 23) ** 2))
    raw_img += detector_noise
    return raw_img


def dark_current(image, current, exposure_time=1.0, gain=1.0, hot_pixels=False):

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


def run_mlem(rawim, mask):
    iguess = np.ones_like(rawim) * 0.5
    guess = iguess
    for i in range(5):
        forward = scipy.signal.convolve2d(guess, mask, "same")
        relative_diff = rawim / (forward + (np.ones_like(forward) * 1e-7))
        back = scipy.signal.correlate2d(relative_diff, mask, "same")
        guess = np.multiply(guess, back)
    return guess


def plot_strahl(signal, sigma_fit, A_fit, central_bins, bin_plot, fluxes, results_tag):
    cmap = cmocean.cm.thermal
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.8))
    im = ax1.imshow(signal, cmap=cmap)
    ax1.axis("off")
    cbar = fig.colorbar(im, ax=ax1, orientation="horizontal", fraction=0.047, pad=0.01)
    cbar.set_label(r"Fluence [cm$^{-2}$ sr$^{-1}$]", fontsize=8)
    cbar.ax.xaxis.labelpad = 1.2
    cbar.ax.tick_params(axis="x", labelsize=8)

    ax2.plot(central_bins, gaussian(central_bins, sigma_fit, A_fit), color="#D57965")
    ax2.plot(bin_plot, fluxes, color="#39329E", label="with PCFOV")

    # ax2.legend(loc="upper right")
    ax2.set_ylabel(r"Fluence [cm$^{-2}$ sr$^{-1}$]", fontsize=8)
    ax2.set_xlabel("Pitch Angle [deg]", fontsize=8)
    ax2.set_yscale("log")
    ax2.xaxis.labelpad = 1.2
    ax2.yaxis.labelpad = 1.2
    ax2.set_xlim([0, 45])
    # ax2.set_ylim([1e6, 1.05e7])

    ax2.tick_params(axis="both", labelsize=8)

    plt.savefig(
        f"../simulation-results/strahl-sweep/{results_tag}_distribution.png",
        dpi=500,
        bbox_inches="tight",
        pad_inches=0.02,
    )


def get_plane_size(distance, det_size_cm, mask_plane_distance_mm, theta_fcfov_deg):
    mask_detector_distance_mm = distance * 10
    plane_distance_to_detector_mm = mask_detector_distance_mm + mask_plane_distance_mm

    detector_diagonal_mm = det_size_cm * 10 * np.sqrt(2)
    detector_half_diagonal_mm = detector_diagonal_mm / 2

    additional_diagonal_mm = (
        np.tan(np.deg2rad(theta_fcfov_deg)) * plane_distance_to_detector_mm
    )

    plane_diagonal_mm = (additional_diagonal_mm + detector_half_diagonal_mm) * 2

    plane_side_length_mm = plane_diagonal_mm / np.sqrt(2)

    # geant asks for half side length

    plane_half_side_length_mm = plane_side_length_mm / 2
    return plane_half_side_length_mm


def gaussian(x, sigma, A):
    halo = 104 * 10**3 * 0.17 * int_time
    return halo + A * np.exp(-((x - 0) ** 2) / (2 * sigma**2))


def polynomial_function(x, *coefficients):
    return np.polyval(coefficients, x)


def fwhm_bins(det_size_cm, params, pixel_size, distance):
    pixel_size = (pixel_size / 3) / 10  # convert to cm
    fwhm_step = 0
    max_rad_dist = np.sqrt(2) * det_size_cm / 2
    bins = []
    bin_sizes = []
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
        bin_sizes.append(bin_size)
        bins.append(angle2)
    bins.insert(0, 0)
    bins = bins[:-1]
    bin_sizes.append(bins[1] - bins[0])
    print("bins", bins)
    return max(bin_sizes)


def get_gf_grid(params, grid_size, geom_factor):
    center = (grid_size - 1) / 2  # Center pixel in a 0-indexed grid
    # Create a meshgrid representing the X and Y coordinates of each pixel
    x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    # Calculate the radial distance from the center for each pixel
    radial_distance = np.sqrt(((x - center) / 3) ** 2 + ((y - center) / 3) ** 2)
    # now i have radiatl distance, use the FWHM thing
    fwhm_grid = polynomial_function(radial_distance, *params)
    fwhm_grid = 2 - (fwhm_grid / np.min(fwhm_grid))
    gf_grid = geom_factor * fwhm_grid / np.sum(fwhm_grid)
    return gf_grid


def flat_field_interp(txt_folder, rank, px_int):
    zz = np.loadtxt(f"{txt_folder}xy-fwhm.txt")
    zc = np.loadtxt(f"{txt_folder}0-fwhm.txt")
    zc = float(zc)
    zz = np.insert(zz, 0, zc)
    reverse_zz = np.flip(zz)
    stacked_zz = np.hstack((reverse_zz, zz[1:]))
    main_diagonal = [[i, i] for i in range(0, (rank * 3 // 2), px_int)]
    main_diagonal = main_diagonal[:-1]
    main_diagonal = np.vstack((-1 * np.flip(main_diagonal[1:]), main_diagonal))

    diagonal_radial = np.array(
        [np.sqrt(md[0] ** 2 + md[1] ** 2) for md in main_diagonal]
    )

    # okay now we have the function
    def polynomial_function(x, *coefficients):
        return np.polyval(coefficients, x)

    # Fit the curve with a polynomial of degree 3 (you can adjust the degree)
    degree = 2
    initial_guess = np.ones(degree + 1)  # Initial guess for the polynomial coefficients
    params, covariance = curve_fit(
        polynomial_function, diagonal_radial, stacked_zz, p0=initial_guess
    )
    return params


def rebuild():
    os.chdir("../EPAD_geant4/build")
    os.system("make")
    os.chdir("../../detector_analysis/")


def run_fwhm(
    data_folder,
    results_folder,
    pix_int,
    incs,
    niter,
    distance,
    n_elements_original,
    det_size_cm,
    pixel_mm,
    txt,
    simulate,
):
    pixel_mm = pixel_mm / 3
    hitsonly = False
    include_hits_effect = True

    sect = 7

    scale = 1
    center_hits = None
    for direction in ["0", "xy"]:
        fwhms = []
        signals = []
        if direction != "0":
            if include_hits_effect:
                hits = np.loadtxt(f"{results_folder}{direction}-hits.txt")
            for ii, inc in enumerate(incs[:-1]):
                if include_hits_effect:
                    hit_norm = hits[ii] / center_hits
                else:
                    hit_norm = 1

                avg_fwhm = 0
                avg_signal = 0
                for i in range(niter):
                    signal, fwhm, _ = run_geom_corr(
                        inc,
                        direction,
                        i,
                        results_folder,
                        simulate=simulate,
                        txt=txt,
                        hitsonly=hitsonly,
                        scale=(scale * hit_norm),
                        data_folder=data_folder,
                        distance=distance,
                        n_elements_original=n_elements_original,
                        det_size_cm=det_size_cm,
                        pixel_mm=pixel_mm,
                        sect=sect,
                    )
                    avg_fwhm += fwhm
                    avg_signal += signal

                fwhms.append(avg_fwhm / niter)
                signals.append(avg_signal / niter)
            # save results
            np.savetxt(
                f"{results_folder}{direction}-fwhm.txt",
                np.array(fwhms),
                delimiter=", ",
                fmt="%.14f",
            )
            np.savetxt(
                f"{results_folder}{direction}-signal.txt",
                np.array(signals),
                delimiter=", ",
                fmt="%.14f",
            )
            print("processed hits for direction ", direction)

        else:
            fwhm_norm = 0
            max_signal_norm = 0
            for i in range(niter):
                max_signal, fwhm, _ = run_geom_corr(
                    0,
                    direction,
                    i,
                    results_folder,
                    simulate=simulate,
                    txt=txt,
                    hitsonly=hitsonly,
                    scale=None,
                    data_folder=data_folder,
                    distance=distance,
                    n_elements_original=n_elements_original,
                    det_size_cm=det_size_cm,
                    pixel_mm=pixel_mm,
                    sect=sect,
                )

                fwhm_norm += fwhm
                max_signal_norm += max_signal

            avg_fwhm_norm = fwhm_norm / niter
            avg_max_signal_norm = max_signal_norm / niter

            scale = avg_max_signal_norm

            np.savetxt(
                f"{results_folder}{direction}-fwhm.txt",
                np.array([avg_fwhm_norm]),
                delimiter=", ",
                fmt="%.14f",
            )
            np.savetxt(
                f"{results_folder}{direction}-signal.txt",
                np.array([avg_max_signal_norm]),
                delimiter=", ",
                fmt="%.14f",
            )
            print("processed center hits with scale ", scale)

            if include_hits_effect:
                center_hits = np.loadtxt(f"{results_folder}{direction}-hits.txt")


def run_ff_hits(
    data_folder,
    results_folder,
    pix_int,
    incs,
    niter,
    distance,
    n_elements_original,
    det_size_cm,
    pixel_mm,
    txt,
    simulate,
):
    pixel_mm = pixel_mm / 3
    hitsonly = True
    for direction in ["0", "xy"]:
        allhits = []
        if direction != "0":
            for inc in incs:
                avg_hits = 0
                for i in range(niter):
                    _, _, nhits = run_geom_corr(
                        inc,
                        direction,
                        i,
                        results_folder,
                        simulate=simulate,
                        txt=txt,
                        hitsonly=hitsonly,
                        data_folder=data_folder,
                        distance=distance,
                        n_elements_original=n_elements_original,
                        det_size_cm=det_size_cm,
                        pixel_mm=pixel_mm,
                    )
                    avg_hits += nhits
                allhits.append(avg_hits / niter)
            np.savetxt(
                f"{results_folder}{direction}-hits.txt",
                np.array(allhits),
                delimiter=", ",
                fmt="%.14f",
            )
        else:
            avg_hits = 0
            for i in range(niter):
                _, _, nhits = run_geom_corr(
                    0,
                    direction,
                    i,
                    data_folder,
                    simulate=simulate,
                    txt=txt,
                    hitsonly=hitsonly,
                    data_folder=data_folder,
                    distance=distance,
                    n_elements_original=n_elements_original,
                    det_size_cm=det_size_cm,
                    pixel_mm=pixel_mm,
                )
                avg_hits += nhits
            center_hits = avg_hits / niter
            np.savetxt(
                f"{results_folder}{direction}-hits.txt",
                np.array([center_hits]),
                delimiter=", ",
                fmt="%.14f",
            )


def modify_file(file_path, line_number, new_content):
    with open(file_path, "r+") as file:
        lines = file.readlines()
        if 0 < line_number <= len(lines):
            lines[line_number - 1] = new_content + "\n"  # Modify the specific line
            file.seek(0)
            file.writelines(lines)
            file.truncate()
        else:
            print("Invalid line number")


def get_hits(fname):
    nthreads = 14
    for hi in range(nthreads):
        fname_hits = fname[:-4] + "-{}.csv".format(hi)
        print(fname_hits)
        myhits = Hits(fname=fname_hits, experiment=False, txt_file=txt)
        hits_dict, sec_brehm, sec_e = myhits.get_det_hits(
            remove_secondaries=True, second_axis="y", det_thick_cm=0.03
        )
        if hi != 0:
            # update fields in hits dict
            myhits.hits_dict["Position"].extend(hits_copy.hits_dict["Position"])
            myhits.hits_dict["Energy"].extend(hits_copy.hits_dict["Energy"])

            hits_copy = copy.copy(myhits)
        else:
            hits_copy = copy.copy(myhits)
    return myhits


# okay so first up, we'll need to run a few geometric factors
simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=True)
nthreads = 14

# rank, and distance!
ranks = [
    23,
]
distances = [3.9]  # cm

mosaic = True
trim = None
holes_inv = True
generate_files = True
check_plots = True

det_size_cm = 5
thickness_um = 100  # um
energy_type = "Mono"
energy_level = 0.235  # keV
det_thick = 300  # um

pyramid_height = 1.1534
rank = 23
distance = 3.9

di = 3

# -------------------------GENERATE PATTERN-------------------------------
gen_pattern = False

n_elements = (2 * rank) - 1
pixel_size_mm = round(10 * det_size_cm / rank, 4)  # mm
# recalcualte detector size
det_size_cm = pixel_size_mm * rank / 10
mask_size_mm = round(pixel_size_mm * (2 * rank - 1), 2)

# okay design generated. Calculate Radii
detector_diagonal_mm = det_size_cm * 10 * np.sqrt(2)
mask_diagonal_mm = mask_size_mm * np.sqrt(2)

detector_half_diagonal_mm = detector_diagonal_mm / 2
mask_half_diagonal_mm = mask_diagonal_mm / 2

# FCFOV half angle
theta_fcfov_deg = np.rad2deg(
    np.arctan((mask_half_diagonal_mm - detector_half_diagonal_mm) / (distance * 10))
)
print(mask_size_mm, theta_fcfov_deg)

fov = np.rad2deg(
    np.arctan(
        (detector_diagonal_mm + (mask_half_diagonal_mm - detector_half_diagonal_mm))
        / (distance * 10)
    )
)
pcfov = fov - theta_fcfov_deg
# calculate spehre center
sphere_center = 1111 * 0.45 - (distance / 2)  # cm

# radius
radius = np.sqrt(
    ((mask_size_mm + 2) * np.sqrt(2) / 2) ** 2
    + ((10 * distance / 2) + pyramid_height) ** 2
)
radius = round(radius / 10, 2)  # cm

txt = True

n_particles = 1e8

results_dir = "../simulation-results/geom-factor/"
fname_tag = f"{rank}_3_gf"
results_tag = f"{fname_tag}"
results_save = results_dir + results_tag
fname = f"{results_save}_raw.txt"
myhits = Hits(fname=fname, experiment=False, txt_file=txt)

geometric_factor = (
    4
    * np.pi**2
    * radius**2
    * np.sum(np.loadtxt(results_save + "_raw.txt"))
    / n_particles
)
print("GEOM", geometric_factor)

data_folder = f"../simulation-data/{rank}-3-fwhm/"
results_folder = f"../simulation-results/{rank}-3-fwhm/"

fwhm_params = flat_field_interp(results_folder, rank, 2)
print(fwhm_params, geometric_factor)
fwhm_params = [2.15056450e-04, -5.78190685e-03, 3.30086919e00]
gf_grid = get_gf_grid(fwhm_params, rank * 3, geometric_factor)
max_bin = fwhm_bins(det_size_cm, fwhm_params, pixel_size_mm, distance)

# -------------------------RUNNING STRAHL-------------------------------
simulate_strahl = True

# set A_fit
int_time = 1
A_fit = 1e6 * int_time

# get size of plane
mask_plane_distance_mm = 5
plane_size_mm = get_plane_size(
    distance, det_size_cm, mask_plane_distance_mm, theta_fcfov_deg
)
plane_size_cm = plane_size_mm / 10
plane_location_cm = (
    (1111 * 0.45) - distance - (thickness_um * 1e-4 / 2) - (mask_plane_distance_mm / 10)
)

sigma_fit = 30
reductions = [0, 1, 0.75, 0.5, 0.25]
reductions = [0]
arrays = []
# simulate smooth bins for now
thetas = np.arange(0, fov, max_bin)
bins = thetas
print(thetas)
txt = False
all_hits = np.zeros((3 * rank, 3 * rank))

for nn, theta in enumerate(thetas[1:]):
    theta_lower = thetas[nn]

    n_particles = int(
        (plane_size_cm * 2) ** 2
        * gaussian(((theta + theta_lower) / 2), sigma_fit, A_fit)
        * 2
        * np.pi
        * (np.cos(np.deg2rad(theta_lower)) - np.cos(np.deg2rad(theta)))
    )
    simulation_engine.set_config(
        det1_thickness_um=300,
        det_gap_mm=30,  # gap between first and second (unused detector)
        win_thickness_um=5,
        det_size_cm=det_size_cm,
        n_elements=n_elements,
        mask_thickness_um=thickness_um,
        mask_gap_cm=distance,
        element_size_mm=pixel_size_mm,
        mosaic=mosaic,
        mask_size=mask_size_mm,
        radius_cm=1,
    )
    if theta > 40:
        theta = theta - 0.01
    formatted_theta = "{:.0f}p{:02d}".format(int(theta), int((theta % 1) * 100))
    fname_tag = f"{rank}-3-{formatted_theta}-deg-{sigma_fit}"

    fname = f"../simulation-data/strahl-sweep/{fname_tag}.csv"
    simulation_engine.set_macro(
        n_particles=int(n_particles),
        energy_keV=[energy_type, energy_level, None],
        surface=True,
        progress_mod=int(n_particles / 10),  # set with 10 steps
        fname_tag=fname_tag,
        confine=False,
        detector_dim=det_size_cm,
        theta=theta,
        theta_lower=theta_lower,
        plane_size_cm=plane_size_cm,
        ring=True,
        sphere_center=plane_location_cm,
        # radius_cm=3,
    )
    if simulate_strahl:
        simulation_engine.run_simulation(fname, build=False, rename=True)
    results_dir = "../simulation-results/strahl-sweep/"
    results_tag = f"{fname_tag}"
    results_save = results_dir + results_tag

    if txt:
        fname = f"{results_save}_raw.txt"
        print(fname)
        myhits = Hits(fname=fname, experiment=False, txt_file=txt)
    else:
        myhits = get_hits(fname)
    deconvolver = Deconvolution(myhits, simulation_engine)

    deconvolver.deconvolve(
        downsample=int(rank * 3),
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
    txt_fname = f"{results_save}_raw.txt"
    # if theta > 35:
    #    all_hits += np.loadtxt(txt_fname) * reduction
    # else:
    #    all_hits += np.loadtxt(txt_fname)
    all_hits += np.loadtxt(txt_fname)

strahl_fname = f"{results_dir}strahl_{rank}_{int_time}_{sigma_fit}_raw.txt"
np.savetxt(
    strahl_fname,
    all_hits,
)

# PROCESS RESULTS
txt = True
myhits = Hits(fname=strahl_fname, experiment=False, txt_file=txt)

# deconvolution steps
deconvolver = Deconvolution(myhits, simulation_engine)

deconvolver.deconvolve(
    downsample=int(rank * 3),
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
bins_ids = {f"{key}": [] for key in range(len(bins) - 1)}
gf_ids = {f"{key}": [] for key in range(len(bins) - 1)}

signal = deconvolver.deconvolved_image / 9

mask, _ = make_mosaic_MURA(
    23,
    0.217,
    holes=False,
    generate_files=False,
)
mask = mask[11:34, 11:34]
mask = np.repeat(
    np.repeat(mask, 3, axis=1),
    3,
    axis=0,
)
plt.clf()
plt.imshow(mask)
plt.savefig("mask.png")
# signal = run_mlem(np.loadtxt(fname), mask)

pixel_count = rank * 3
center_pixel = pixel_count // 2
pixel_size = (pixel_size_mm / 3) / 10

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

# finally get the results ready to plot
fluxes = []
bin_plot = []
central_bins = []
fd = []
ff_save = []
for ii, bn in enumerate(bins[:-1]):
    flux = np.sum(np.array(bins_ids[f"{ii}"])) / np.sum(np.array(gf_ids[f"{ii}"]))
    flux_diff = gaussian((bins[ii + 1] + bn) / 2, sigma_fit, A_fit) - flux
    # if reduction != 0:
    #    flux_diff = fd_save[ii] - flux
    #    fd.append(np.absolute(100 * (flux_diff / fd_save[ii])))
    central_bins.append((bins[ii + 1] + bn) / 2)
    bin_plot.append(bn)
    bin_plot.append(bins[ii + 1])
    fluxes.append(flux)
    fluxes.append(flux)

    ff_save.append(flux)
# if reduction == 0:
#    fd_save = ff_save
# else:
#    print("reduction", reduction, "average", np.average(np.array(fd[:8])))
central_bins = np.array(central_bins)

pcfov_save = f"{results_dir}strahl_{rank}_{int_time}_{sigma_fit}_fcfov.txt"
# np.savetxt(pcfov_save, np.array(fluxes))
# no_pcfov_flux = np.loadtxt(pcfov_save)

plot_strahl(
    signal,
    sigma_fit,
    A_fit,
    central_bins,
    bin_plot,
    fluxes,
    results_tag + "window",
)
arrays.append(signal)


"""
fig, axs = plt.subplots(1, 4, figsize=(10, 3))
cmap = cmocean.cm.thermal
for i in range(4):
    im = axs[i].imshow(arrays[i], cmap=cmap)
    cb = fig.colorbar(
        im,
        ax=axs[i],
        orientation="horizontal",
        pad=0.01,
    )
    axs[i].axis("off")
    axs[i].set_title(f"{round(100*(1-reductions[i]))}% reduction")
    cb.set_label(r"Fluence [cm$^{-2}$ sr$^{-1}$]", fontsize=8)
    cb.ax.tick_params(axis="x", labelsize=8)

plt.savefig(
    "../simulation-results/strahl-sweep/PCFOV.png",
    dpi=500,
    bbox_inches="tight",
    pad_inches=0.02,
)
"""
