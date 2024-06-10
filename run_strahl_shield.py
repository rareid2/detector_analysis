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
from plotting.plot_geant_histograms import *
from functools import partial
from scipy.optimize import minimize


def make_pa(signal, gf_grid):
    bins = np.arange(0, fov, 3.12)  # 3.2 deg resolution
    thetas = bins
    rank = 23
    pixel_size_mm = 1.7
    pixel_count = rank * 3
    center_pixel = pixel_count // 2
    pixel_size = (pixel_size_mm / 3) / 10
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

            for ii, bn in enumerate(bins[:-1]):
                if angle >= bn and angle < bins[ii + 1]:
                    bins_ids[f"{ii}"].append(signal[y, x])
                    gf_ids[f"{ii}"].append(gf_grid[y, x])
                if ii == len(bins[:-1]):
                    if angle >= bn and angle <= bins[ii + 1]:
                        bins_ids[f"{ii}"].append(signal[y, x])
                        gf_ids[f"{ii}"].append(gf_grid[y, x])
    ff_save = []
    central_bins = []
    for ii, bn in enumerate(bins[:-1]):

        central_bin = (bins[ii + 1] + bn) / 2
        flux = np.sum(np.array(bins_ids[f"{ii}"])) / np.sum(np.array(gf_ids[f"{ii}"]))
        # true_flux = gaussian(central_bin, sigma_fit, A_fit, halo)

        central_bins.append(central_bin)
        ff_save.append(flux)
    return ff_save, central_bins


def objective_func(params, signal):
    (coeff1, coeff2, geometric_factor) = params

    gf_grid = get_gf_grid(rank * 3, geometric_factor, [coeff1, coeff2, 1])
    flux, central_bins = make_pa(signal, gf_grid)
    # get truth
    int_time = 50
    sigma_fit = 8.5
    A_fit = 1e6 * int_time
    halo = 104 * 10**3 * 0.17 * int_time
    true_flux = gaussian(np.array(central_bins), sigma_fit, A_fit, halo)

    true_flux = np.array(true_flux[:13])
    flux = np.array(flux[:13])

    min_me = true_flux - flux
    result = np.average(np.absolute(min_me))
    return result


def plot_strahl(signal, sigma_fit, A_fit, central_bins, fluxes, results_tag, halo):
    cmap = cmocean.cm.thermal
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.8))
    im = ax1.imshow(signal, cmap=cmap)
    ax1.axis("off")

    cbar = fig.colorbar(im, ax=ax1, orientation="horizontal", fraction=0.047, pad=0.01)
    cbar.set_label("Signal", fontsize=8)
    cbar.ax.xaxis.labelpad = 1.2
    cbar.ax.tick_params(axis="x", labelsize=8)
    cb = np.linspace(0, 180, 100)

    ax2.plot(
        cb,
        gaussian(cb, sigma_fit, A_fit, halo),
        color="#D57965",
        label="True pitch angle distribution",
    )
    ax2.scatter(
        central_bins,
        fluxes,
        color="#39329E",
        label="Observations",
    )

    partial_model_func = partial(gaussian, extra_param=halo)
    initial_guess = [sigma_fit, A_fit]
    params, cov = curve_fit(
        partial_model_func,
        central_bins,
        fluxes,
        p0=initial_guess,
        maxfev=100000,
    )
    print(params)

    ax2.plot(
        cb,
        gaussian(cb, params[0], params[1], halo),
        "--",
        color="#39329E",
        label="Observed pitch angle distribution",
    )
    ax2.text(
        60,
        0.12 * A_fit,
        rf"True strahl FWHM = {round(sigma_fit*2.355,1)}$^\circ$",
        fontsize=8,
    )
    ax2.text(
        60,
        0.1 * A_fit,
        rf"Observed strahl FWHM = {round(params[0]*2.355,1)}$^\circ$",
        fontsize=8,
    )
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_ylabel(r"Fluence [cm$^{-2}$ sr$^{-1}$]", fontsize=8)
    ax2.set_xlabel("Pitch Angle [deg]", fontsize=8)
    ax2.set_yscale("log")
    ax2.xaxis.labelpad = 1.2
    ax2.yaxis.labelpad = 1.2
    ax2.set_xlim([0, 180])
    # ax2.set_ylim([5e4, 1.2e7])

    ax2.tick_params(axis="both", labelsize=8)

    plt.savefig(
        f"../simulation-results/strahl-sweep/distributions/{results_tag}_distribution.png",
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


def gaussian(x, sigma, A, extra_param):
    return extra_param + A * np.exp(-((x - 0) ** 2) / (2 * sigma**2))


def polynomial_function(x, *coefficients):
    return np.polyval(coefficients, x)


def get_gf_grid(grid_size, geom_factor, coefficients):
    center = (grid_size - 1) / 2  # Center pixel in a 0-indexed grid
    # Create a meshgrid representing the X and Y coordinates of each pixel
    x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    # Calculate the radial distance from the center for each pixel
    radial_distance = np.sqrt(((x - center) / 3) ** 2 + ((y - center) / 3) ** 2)
    # coefficients
    fwhm_grid = np.polyval(coefficients, radial_distance)
    fwhm_grid = fwhm_grid / np.sum(fwhm_grid)
    # fwhm_grid = polynomial_function(radial_distance, *params)
    # fwhm_grid = 2 - (fwhm_grid / np.min(fwhm_grid))
    gf_grid = geom_factor * fwhm_grid

    return gf_grid


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
    energy_level_keV,
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
                        energy_level_keV=energy_level_keV,
                    )
                    avg_hits += nhits
                allhits.append(avg_hits / niter)
            np.savetxt(
                f"{results_folder}{direction}-{energy_level_keV}-hits-no-egg.txt",
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
                    energy_level_keV=energy_level_keV,
                )
                avg_hits += nhits
            center_hits = avg_hits / niter
            np.savetxt(
                f"{results_folder}{direction}-{energy_level_keV}-hits.txt",
                np.array([center_hits]),
                delimiter=", ",
                fmt="%.14f",
            )


def get_hits(fname):
    nthreads = 12
    for hi in range(nthreads):
        fname_hits = fname[:-4] + "-{}.csv".format(hi)
        print(fname_hits)
        myhits = Hits(fname=fname_hits, experiment=False, txt_file=txt)
        hits_dict, sec_e, sec_brehm = myhits.get_det_hits(
            remove_secondaries=False, second_axis="y", det_thick_cm=0.005
        )
        if hi != 0:
            # update fields in hits dict
            myhits.hits_dict["Position"].extend(hits_copy.hits_dict["Position"])
            myhits.hits_dict["Energy"].extend(hits_copy.hits_dict["Energy"])
            myhits.hits_dict["Vert"].extend(hits_copy.hits_dict["Vert"])
            myhits.hits_dict["E0"].extend(hits_copy.hits_dict["E0"])
            myhits.hits_dict["Edep"].extend(hits_copy.hits_dict["Edep"])

            hits_copy = copy.copy(myhits)
        else:
            hits_copy = copy.copy(myhits)
    return myhits


# ---------------------------------------------------------------------------------------------------------------------------------
simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=True)
nthreads = 14

# rank, and distance!
rank = 23
distance = 3.12  # 1.58  # cm

mosaic = True
trim = None

det_size_cm = 3.91  # 2.047
thickness_um = 100  # um # CODED AP THICK

energy_type = "Pow"
energy_min = 0.1
energy_max = 3

det_thick = 50  # um

pyramid_height = 1  # unused

n_elements = (2 * rank) - 1
pixel_size_mm = 1.7  # 0.89  # mm - for now just look at it as 10 microns
# recalcualte detector size
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

fov = np.rad2deg(
    np.arctan(
        (detector_diagonal_mm + (mask_half_diagonal_mm - detector_half_diagonal_mm))
        / (distance * 10)
    )
)
pcfov = fov - theta_fcfov_deg
print("THE PCFOV", pcfov)
# calculate spehre center
sphere_center = 1111 * 0.45 - (distance / 2)  # cm

# radius
radius = np.sqrt(
    ((mask_size_mm + 2) * np.sqrt(2) / 2) ** 2
    + ((10 * distance / 2) + pyramid_height) ** 2
)
radius = round(radius / 10, 2)  # cm

n_inst_bins = 18
instrument_bins = np.logspace(
    np.log10(energy_min), np.log10(energy_max), n_inst_bins + 1
)

# lets do JUST hits
data_folder = (
    f"/home/rileyannereid/workspace/geant4/simulation-data/{rank}-3-fwhm-egg-big/"
)
results_folder = f"../simulation-results/{rank}-3-fwhm-egg-big/"
txt = False
simulate = True
n_elements_original = 23
pixel_mm = 1.7

pix_int = 3
niter = 3
incs = np.arange(0, 60, pix_int)

central_energy = 0.300
central_energy = round(central_energy, 2)
"""
run_ff_hits(
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
    central_energy,
)


txt = False
n_particles = 1e8

pcfov_count = 0
fcfov_count = 0
gfs = []

for i in range(1):
    results_dir = "../simulation-results/geom-factor/"
    fname_tag = f"{rank}_egg_gf_big_{i}"

    results_tag = f"{fname_tag}"
    results_save = results_dir + results_tag
    fname = f"../simulation-data/geom-factor/{fname_tag}_{n_particles:.2E}.csv"

    simulation_engine.set_config(
        det1_thickness_um=det_thick,
        det_gap_mm=30,  # gap between first and second (unused detector)
        win_thickness_um=100,  # window is not actually in there
        det_size_cm=det_size_cm,
        n_elements=n_elements,
        mask_thickness_um=thickness_um,
        mask_gap_cm=distance,
        element_size_mm=pixel_size_mm,
        mosaic=mosaic,
        mask_size=mask_size_mm,
        radius_cm=radius,
        vol_name=f"lens_pyramid_1",
    )

    simulation_engine.set_macro(
        n_particles=int(n_particles),
        energy_keV=[energy_type, energy_min, energy_max],
        surface=True,
        progress_mod=int(n_particles / 10),  # set with 10 steps
        fname_tag=fname_tag,
        confine=False,
        detector_dim=det_size_cm,
        theta=None,
        ring=False,
        radius_cm=radius,
        sphere_center=sphere_center,
    )
    #simulation_engine.run_simulation(fname, build=True, rename=True)

    myhits = get_hits(fname)

    # find angle between plane and 3D vector
    for pos, p0 in zip(myhits.hits_dict["Position"], myhits.hits_dict["Vert"]):
        # find vector
        pz = 499.948
        v0 = np.array([pos[0] - p0[0], pos[1] - p0[1], pz - p0[2]])
        # z axis
        vz = np.array([0, 0, 1])
        vtheta = np.arccos(np.dot(v0, vz) / (np.linalg.norm(v0)))
        vtheta_deg = np.rad2deg(vtheta)

        if 41 < vtheta_deg:
            pcfov_count += 1
            if vtheta_deg > 69:
                print("weird", vtheta_deg)
        else:
            fcfov_count += 1

    print(pcfov_count / (pcfov_count + fcfov_count))

    geometric_factor = (
        4 * np.pi**2 * radius**2 * len(myhits.hits_dict["Position"]) / n_particles
    )

    gfs.append(geometric_factor)

# get average GF
gfs = np.array(gfs)
print("all", gfs)
print(np.std(gfs), np.average(gfs))

# now we sort by energy
histo_dir = "/home/rileyannereid/workspace/geant4/simulation-data/energy-spectrum"
histo_results_dir = (
    "/home/rileyannereid/workspace/geant4/simulation-results/energy-spectrum"
)

fname_tag = f"src_spectrum"
i = 1
k = 1
with open(f"{histo_dir}/{fname_tag}_h%d_h%d.%d.csv" % (i, i, k)) as f:
    lines = [line for line in f]
# convert histogram to data
histo = convert_from_csv(lines, fname_tag)

figure = plt.figure()

bins = np.array(histo["bins"][0])
x = (histo["bin_edges"][0] + histo["bin_edges"][1]) / 2
xerr = (histo["bin_edges"][0] - histo["bin_edges"][1]) / 2

y = bins[:, 1]
_entries = np.sum(bins[:, 0])
yerr = np.sqrt(bins[:, 2] - bins[:, 1] ** 2 / _entries)
plt.xlim(x[0] - xerr[0], x[-1] + xerr[-1])
plt.plot(x, y, markersize=3)

fname = f"{histo_results_dir}/source_spectrum.png"
plt.savefig(fname)
plt.clf()

simulated_bins = histo["bin_edges"][0]
simulated_bins = np.append(simulated_bins, energy_max)

# for each incident particle - where did it deposut energy?
import cmocean
cmap = cmocean.cm.thermal

counts = np.zeros((len(instrument_bins), len(x)))
total_gf = 0
# actual gf per bin
gf_energy = np.zeros((18))
sim_energy = np.zeros((18))
for hit_n, hit in enumerate(myhits.hits_dict["E0"]):
    # find which y this is at
    for si, sb in enumerate(simulated_bins[:-1]):
        if sb <= hit < simulated_bins[si + 1]:
            n_sim = si

    # now we know which n_sim, what energy did it deposit?
    hit_energy = myhits.hits_dict["Edep"][hit_n]
    # now we know which simulated bin it is in
    # which instrument bin does that fall into???
    for bn, ib in enumerate(instrument_bins[:-1]):
        if ib <= hit_energy < instrument_bins[bn + 1]:
            # now we have the right instrument bin to include it in
            counts[bn, n_sim] += 1
            gf_energy[bn] += 1
    if 0.1 <= hit_energy <= 3:
        total_gf += 1

for xi, x_sim in enumerate(x):
    for bn, ib in enumerate(instrument_bins[:-1]):
        if ib <= x_sim < instrument_bins[bn + 1]:
            sim_energy[bn] += y[xi]

gf_factor = 4 * np.pi**2 * radius**2
print("total instrument gf", gf_factor * total_gf / 1e8)
for yi, yy in enumerate(counts[:-1]):
    linex = []
    liney = []
    for xi, xx in enumerate(x):
        linex.append(simulated_bins[xi])
        linex.append(simulated_bins[xi + 1])
        liney.append((yy[xi] / y[xi]) * gf_factor)
        liney.append((yy[xi] / y[xi]) * gf_factor)
    plt.plot(
        linex,
        liney,
        color=cmap(yi / n_inst_bins),
        label=f"{round(instrument_bins[yi],2)}-{round(instrument_bins[yi+1],2)} keV",
    )
plt.legend(fontsize="8", ncol=2)
plt.xlabel("Incident Energy [keV]")
plt.ylabel("Geometric Factor [cm^2 sr]")
plt.savefig(f"{histo_results_dir}/gf_dep_egg_big_{i}.png", dpi=500)
plt.clf()

"""
plot_gf = False
sigma_fits = [8.5, 29.7]
int_time = 10
results_dir = "../simulation-results/strahl-sweep/"
for li, ly in enumerate(instrument_bins[:1]):
    all_energy = []
    if li == 8 or li == 0:
        for sigma_fit in sigma_fits[-1:]:
            signals = []

            central_energy = (instrument_bins[li] + instrument_bins[li + 1]) / 2

            """
            if plot_gf:
                plt.clf()

                fig, axs = plt.subplots(1, 1, figsize=(4, 4))
                im = plt.imshow(gf_grid, cmap=cmap)
                cbar1 = fig.colorbar(
                    im, orientation="horizontal", pad=0.01, shrink=0.84
                )
                cbar1.set_label(r"Geometric Factor [cm$^2$ sr]", fontsize=10)
                axs.axis("off")
                plt.title(
                    f"{round(instrument_bins[li],2)} keV - {round(instrument_bins[li+1],2)} keV"
                )
                plt.savefig(
                    f"../simulation-results/{results_dir}_gf_grid_{li}",
                    dpi=500,
                    bbox_inches="tight",
                    pad_inches=0.02,
                )
            """

            # simulate smooth bins for now
            bins = np.arange(0, fov, 3.12)  # 3.2 deg resolution
            thetas = bins

            # set A_fit
            if li == 0:
                A_fit = 1e6 * int_time
                halo = 104 * 10**3 * 0.17 * int_time
            elif li == 8:
                A_fit = 6.6e4 * int_time
                halo = 104 * 10**3 * 0.17 * int_time / 15.151515
            else:
                A_fit = 3 * int_time

            # get size of plane
            mask_plane_distance_mm = 5
            plane_size_mm = get_plane_size(
                distance, det_size_cm, mask_plane_distance_mm, theta_fcfov_deg
            )
            plane_size_cm = plane_size_mm / 10
            plane_location_cm = (
                (1111 * 0.45)
                - distance
                - (thickness_um * 1e-4 / 2)
                - (mask_plane_distance_mm / 10)
            )
            all_hits = np.zeros((69, 69))
            for nn, theta in enumerate(thetas[1:-1]):
                theta_lower = thetas[nn]

                n_particles = int(
                    (plane_size_cm * 2) ** 2
                    * gaussian(((theta + theta_lower) / 2), sigma_fit, A_fit, halo)
                    * 2
                    * np.pi
                    * (np.cos(np.deg2rad(theta_lower)) - np.cos(np.deg2rad(theta)))
                )

                simulation_engine.set_config(
                    det1_thickness_um=50,
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

                formatted_theta = "{:.0f}p{:02d}".format(
                    int(theta), int((theta % 1) * 100)
                )
                num_str = f"{theta:.2f}"  # Ensure the number is formatted to two decimal places
                parts = num_str.split(".")
                formatted_theta = parts[0] + "p" + parts[1]

                fname_tag = f"{rank}-3-{li}-{formatted_theta}-deg-{round(sigma_fit)}-big-{int_time}-NOEGG"

                fname = f"../simulation-data/strahl-sweep/{fname_tag}.csv"
                simulation_engine.set_macro(
                    n_particles=int(n_particles),
                    energy_keV=["Mono", central_energy, None],
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

                simulation_engine.run_simulation(fname, build=False, rename=True)

                results_dir = "../simulation-results/strahl-sweep/"
                results_tag = f"{fname_tag}"
                results_save = results_dir + results_tag

                txt = False
                if txt:
                    fname = f"{results_save}_raw.txt"
                    myhits = Hits(fname=fname, experiment=False, txt_file=txt)
                else:
                    print("getting hits")
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
                signals.append(np.sum(deconvolver.deconvolved_image) / 9)
                txt_fname = f"{results_save}_raw.txt"
                all_hits += np.loadtxt(txt_fname)

            strahl_fname = (
                f"{results_dir}strahl_{rank}_{int_time}_{round(sigma_fit)}_raw.txt"
            )
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
            signal = deconvolver.deconvolved_image / 9
            """
            last_dev = float("inf")
            for c1 in np.linspace(0.0005, 0.003, 30):
                for c2 in np.linspace(-0.2, -0.05, 30):
                    for gf in np.linspace(1.5, 2.5, 30):
                        # print(c1, c2, gf)
                        dev = objective_func([c1, c2, gf], signal)
                        if dev <= last_dev:
                            print(c1, c2, gf)
                            last_dev = dev
            """
            gf_grid = get_gf_grid(
                rank * 3,
                2.29,
                [0.0011, -0.082, 1],
            )

            bins_ids = {f"{key}": [] for key in range(len(bins) - 1)}
            gf_ids = {f"{key}": [] for key in range(len(bins) - 1)}

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

            # finally get the results ready to plot!
            fluxes = []
            bin_plot = []
            central_bins = []
            fd = []
            fdiffs = []
            gfs = []
            ff_save = []
            for ii, bn in enumerate(bins[:-1]):

                central_bin = (bins[ii + 1] + bn) / 2
                flux = np.sum(np.array(bins_ids[f"{ii}"])) / np.sum(
                    np.array(gf_ids[f"{ii}"])
                )
                true_flux = gaussian(central_bin, sigma_fit, A_fit, halo)
                flux_diff = (true_flux - flux) / true_flux

                central_bins.append(central_bin)
                fdiffs.append(flux_diff)
                ff_save.append(flux)

                print(
                    "signal",
                    np.sum(np.array(bins_ids[f"{ii}"])),
                    "flux",
                    flux,
                    "ff_diff",
                    flux_diff,
                    "bin",
                    central_bin,
                    "gf",
                    np.sum(np.array(gf_ids[f"{ii}"])),
                )

                gfs.append(np.sum(np.array(gf_ids[f"{ii}"])))

            stop_ind = 12
            central_bins = np.array(central_bins[:stop_ind])

            plt.clf()
            plt.plot(central_bins, signals[:stop_ind])
            plt.savefig(f"signals__{li}_{round(sigma_fit)}.png")
            plt.clf()

            plt.clf()
            plt.plot(central_bins, gfs[:stop_ind])
            plt.savefig(f"gf_{li}_{round(sigma_fit)}.png")
            plt.clf()

            plt.clf()
            plt.plot(
                central_bins, np.array(signals[:stop_ind]) / np.array(gfs[:stop_ind])
            )
            plt.savefig(f"flux_{li}_{round(sigma_fit)}.png")
            plt.clf()

            plot_strahl(
                signal,
                sigma_fit,
                A_fit,
                central_bins,
                ff_save[:stop_ind],
                results_tag,
                halo,
            )
