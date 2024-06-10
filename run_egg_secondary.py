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
import scipy
import cmocean
from plotting.plot_geant_histograms import *


def run_geom_corr(
    ind,
    direction,
    i,
    results_folder,
    vmax=None,
    simulate=False,
    txt=True,
    hitsonly=False,
    scale=None,
    data_folder=None,
    distance=float,
    n_elements_original=int,
    det_size_cm=float,
    pixel_mm=float,
    sect=None,
    e_level_keV=float,
):
    nthreads = 14
    if data_folder is None:
        data_folder = results_folder

    if simulate:
        simulation_engine = SimulationEngine(
            construct="CA", source="PS", write_files=True
        )
    else:
        simulation_engine = SimulationEngine(
            construct="CA", source="PS", write_files=False
        )

    # general detector design
    # det_size_cm = 4.94  # cm
    # pixel = 1.26666667  # mm
    pixel_cm = pixel_mm * 0.1

    # ---------- coded aperture set up ---------
    # set number of elements
    # n_elements_original = 13
    multiplier = 3

    # focal length
    # distance = 4.3  # cm

    # det_size_cm = 2.82 #4.984  # cm
    # pixel = 0.2 #0.186666667  # mm
    # pixel_cm = pixel * 0.1  # cm

    # ---------- coded aperture set up ---------

    # set number of elements
    # n_elements_original = 47 #89
    # multiplier = 3

    element_size = pixel_mm * multiplier
    n_elements = (2 * n_elements_original) - 1

    mask_size = element_size * n_elements
    # no trim needed for custom design
    trim = None
    mosaic = True

    # thickness of mask
    det_thickness = 300  # um
    thickness = 100  # um

    # focal length
    # distance = 2.2 # 4.49  # cm

    fake_radius = 1

    # set distance of the source to the detector
    world_offset = 1111 * 0.45
    detector_loc = world_offset - ((det_thickness / 2) * 1e-4)
    ca_pos = world_offset - distance - (((thickness / 2) + (det_thickness / 2)) * 1e-4)
    src_plane_distance = world_offset + 500

    # pixel location
    pix_ind = ind
    if direction == "x":
        px = pixel_cm * pix_ind
        py = 0
    elif direction == "y":
        px = 0
        py = pixel_cm * pix_ind
    elif direction == "xy":
        px = pixel_cm * pix_ind
        py = pixel_cm * pix_ind
    else:
        px = 0
        py = 0
    pz = detector_loc

    # source plane
    src_plane_normal = np.array([0, 0, 1])  # Normal vector of the plane
    src_plane_pt = np.array([0, 0, -500])  # A point on the plane

    px_point = np.array([px, py, pz])  # A point on the line
    px_ray = np.array([0 - px, 0 - py, ca_pos - pz])

    ndotu = src_plane_normal.dot(px_ray)

    epsilon = 1e-8
    if abs(ndotu) < epsilon:
        print("no intersection or line is within plane")

    w = px_point - src_plane_pt
    si = -src_plane_normal.dot(w) / ndotu
    src_point = w + si * px_ray + src_plane_pt

    # now we have location of the src_point

    # ------------------- simulation parameters ------------------
    if hitsonly:
        n_particles = 1e7
    else:
        n_particles = 1e6

    # --------------set up simulation---------------
    simulation_engine.set_config(
        det1_thickness_um=det_thickness,
        det_gap_mm=30,  # gap between first and second (unused detector)
        win_thickness_um=100,  # window is not actually in there
        det_size_cm=det_size_cm,
        n_elements=n_elements,
        mask_thickness_um=thickness,
        mask_gap_cm=distance,
        element_size_mm=element_size,
        mosaic=mosaic,
        mask_size=mask_size,
        radius_cm=fake_radius,
    )

    # --------------set up source---------------
    energy_type = "Mono"
    energy_level = e_level_keV  # keV

    # --------------set up data naming---------------
    if hitsonly:
        fname_tag = f"hitsonly-{n_elements_original}-{distance}-{ind}-{direction}-{i}_{n_particles:.2E}_{energy_type}_{energy_level}"
    else:
        fname_tag = f"{n_elements_original}-{distance}-{ind}-{direction}-{i}_{n_particles:.2E}_{energy_type}_{energy_level}"

    if txt:
        fname = f"{results_folder}{fname_tag}_raw.txt"
    else:
        fname = f"{data_folder}{fname_tag}.csv"

    simulation_engine.set_macro(
        n_particles=int(n_particles),
        energy_keV=[energy_level],
        surface=False,
        positions=[[src_point[0], src_point[1], -500]],
        directions=[1],
        progress_mod=int(n_particles / 10),  # set with 10 steps
        fname_tag=fname_tag,
        detector_dim=det_size_cm,
    )

    # --------------RUN AND PROCESS---------------
    # directory to save results in
    results_save = f"{results_folder}{fname_tag}"
    # print(np.sum(np.loadtxt(fname)))

    sec_gamma_count = 0
    sec_gamma_in_range = 0
    sec_e_count = 0
    sec_e_in_range = 0
    sec_e = 0
    sec_brehm = 0

    simulation_engine.run_simulation(fname, build=False, rename=True)

    print("PROCESSING CSV")
    for hi in range(nthreads):
        print(hi)
        fname_hits = fname[:-4] + "-{}.csv".format(hi)
        myhits = Hits(fname=fname_hits, experiment=False, txt_file=txt)
        hits_dict, se, sb = myhits.get_det_hits(
            remove_secondaries=False, second_axis="y"
        )
        sec_e += se
        sec_brehm += sb

        if hi != 0:
            # update fields in hits dict
            myhits.hits_dict["Position"].extend(hits_copy.hits_dict["Position"])
            myhits.hits_dict["Energy"].extend(hits_copy.hits_dict["Energy"])
            myhits.hits_dict["Edep"].extend(hits_copy.hits_dict["Edep"])
            myhits.hits_dict["ID"].extend(hits_copy.hits_dict["ID"])
            myhits.hits_dict["name"].extend(hits_copy.hits_dict["name"])

            hits_copy = copy.copy(myhits)
        else:
            hits_copy = copy.copy(myhits)

    for hi_n, hit_id in enumerate(myhits.hits_dict["ID"]):
        if hit_id != 0:
            # secondary particle
            # electron or brehm?
            if myhits.hits_dict["name"][hi_n] == "e-":
                # electron
                sec_e_count += 1
                if 0.1 <= myhits.hits_dict["Edep"][hi_n] <= 3:
                    sec_e_in_range += 1
            elif myhits.hits_dict["name"][hi_n] == "gamma":
                sec_gamma_count += 1
                if 0.1 <= myhits.hits_dict["Edep"][hi_n] <= 3:
                    sec_gamma_in_range += 1

    total_hit_count = len(myhits.hits_dict["name"])

    return (
        total_hit_count,
        sec_e_count,
        sec_e_in_range,
        sec_gamma_count,
        sec_gamma_in_range,
    )


def polynomial_function(x, *coefficients):
    return np.polyval(coefficients, x)


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
    inc = incs[-1]
    avg_hits = 0
    i = 0
    total_electrons = []
    electrons_in_range = []
    total_gammas = []
    gammas_in_range = []

    for e_level_keV in np.logspace(-0.69, 3, 20):
        (
            total_hit_count,
            sec_e_count,
            sec_e_in_range,
            sec_gamma_count,
            sec_gamma_in_range,
        ) = run_geom_corr(
            inc,
            "xy",
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
            e_level_keV=round(e_level_keV, 3),
        )
        hit_factor = (total_hit_count - sec_e_count - sec_gamma_count) / 1000
        total_electrons.append(sec_e_count / hit_factor)
        total_gammas.append(sec_gamma_count / hit_factor)

        electrons_in_range.append(sec_e_in_range / hit_factor)
        gammas_in_range.append(sec_gamma_in_range / hit_factor)

    return total_electrons, total_gammas, electrons_in_range, gammas_in_range


def get_hits(fname):
    nthreads = 14
    for hi in range(nthreads):
        fname_hits = fname[:-4] + "-{}.csv".format(hi)
        print(fname_hits)
        myhits = Hits(fname=fname_hits, experiment=False, txt_file=txt)
        hits_dict, sec_e, sec_brehm = myhits.get_det_hits(
            remove_secondaries=False, second_axis="y", det_thick_cm=0.03
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


simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=True)
nthreads = 14

# rank, and distance!
rank = 23
distance = 3.12  # 1.58  # cm

mosaic = True
trim = None

det_size_cm = 3.91  # 2.047
thickness_um = 100  # um # CODED AP THICK
det_thick = 300  # um

pyramid_height = 1  # unused

n_elements = (2 * rank) - 1
pixel_size_mm = 1.7  # mm - for now just look at it as 10 microns

gf = 2.35
data_folder = f"../simulation-data/{rank}-3-fwhm-egg-big/"
results_folder = f"../simulation-results/{rank}-3-fwhm-egg-big/"
pix_int = 3
niter = 3
incs = range(pix_int, 10, pix_int)
txt = False
simulate = True
n_elements_original = 23
pixel_mm = 1.7

total_electrons, total_gammas, electrons_in_range, gammas_in_range = run_ff_hits(
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
)

# make plot

plt.clf()
plt.figure(figsize=(10, 6))
x = np.logspace(-0.69, 3, 20)

plt.semilogx(x, total_electrons, color="#0C2D5F", label="All secondary e-")
plt.semilogx(
    x,
    electrons_in_range,
    color="#4989E9",
    label="Secondary e- between 100eV and 3keV",
)

plt.semilogx(x, total_gammas, color="#FAA83C", label="All bremsstrahlung photons")

plt.ylabel("Detections per 1000 incident e-")
plt.xlabel("Simulated e- energy [keV]")
plt.legend()
plt.savefig(
    f"../simulation-results/shield_secondaries.png",
    dpi=500,
    bbox_inches="tight",
    pad_inches=0.02,
)
