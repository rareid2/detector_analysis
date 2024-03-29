from simulation_engine import SimulationEngine
from hits import Hits
from scattering import Scattering
from deconvolution import Deconvolution

from macros import find_disp_pos
from scipy.optimize import minimize
import numpy as np
import os

# construct = CA and TD
# source = DS and PS


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
):
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
    det_size_cm = 3.05  # cm
    pixel = 0.25  # mm
    pixel_cm = 0.025

    # ---------- coded aperture set up ---------

    # set number of elements
    n_elements_original = 61
    multiplier = 2

    element_size = pixel * multiplier
    n_elements = (2 * n_elements_original) - 1

    mask_size = element_size * n_elements
    # no trim needed for custom design
    trim = None
    mosaic = True

    # thickness of mask
    det_thickness = 300  # um
    thickness = 400  # um

    # focal length
    distance = 2  # cm

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
    n_particles = 1e6

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
        radius_cm=fake_radius,
    )

    # --------------set up source---------------
    energy_type = "Mono"
    energy_level = 100  # keV

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

    if simulate and not hitsonly:
        simulation_engine.run_simulation(fname, build=True, rename=True)

        # get the raw hits
        myhits = Hits(fname=fname, experiment=False)
        myhits.get_det_hits(
            remove_secondaries=True, second_axis="y", energy_level=energy_level
        )
        hits_len = len(myhits.hits_dict["Position"])

        # process them (txt file will be saved)
        deconvolver = Deconvolution(myhits, simulation_engine)
        deconvolver.deconvolve(
            downsample=int(multiplier * n_elements_original),
            trim=trim,
            vmax=vmax,
            plot_deconvolved_heatmap=True,
            plot_raw_heatmap=True,
            save_raw_heatmap=results_save + "_raw.png",
            save_deconvolve_heatmap=results_save + "_dc.png",
            plot_signal_peak=False,
            plot_conditions=False,
            hits_txt=False,
        )
    elif txt and not hitsonly:
        # dont simulate, process the txt file
        myhits = Hits(fname=fname, experiment=False, txt_file=True)
        hits_len = np.sum(np.loadtxt(fname))

        # process them
        deconvolver = Deconvolution(myhits, simulation_engine)
        deconvolver.deconvolve(
            downsample=int(multiplier * n_elements_original),
            trim=trim,
            vmax=vmax,
            plot_deconvolved_heatmap=False,
            plot_raw_heatmap=False,
            save_raw_heatmap=results_save + "_raw.png",
            save_deconvolve_heatmap=results_save + "_dc.png",
            plot_signal_peak=False,
            plot_conditions=False,
            hits_txt=True,
        )
    else:
        # only getting raw hits
        if simulate:
            simulation_engine.run_simulation(fname, build=True, rename=True)
            myhits = Hits(fname=fname, experiment=False)
            myhits.get_det_hits(
                remove_secondaries=True, second_axis="y", energy_level=energy_level
            )
            hits_len = len(myhits.hits_dict["Position"])
        if txt:
            myhits = Hits(fname=fname, experiment=False, txt_file=True)
            hits_len = np.sum(np.loadtxt(fname))
        else:
            myhits = Hits(fname=fname, experiment=False)
            myhits.get_det_hits(
                remove_secondaries=True, second_axis="y", energy_level=energy_level
            )
            hits_len = len(myhits.hits_dict["Position"])

    if not hitsonly:
        # find the max index
        max_index_flat = np.argmax(deconvolver.deconvolved_image)
        max_index_2d = np.unravel_index(
            max_index_flat, deconvolver.deconvolved_image.shape
        )

        # shift up to the noise floor so that the noise floor is at 0

        deconvolver.shift_noise_floor_ptsrc(max_index_2d[0], max_index_2d[1])
        max_signal = deconvolver.shifted_image[max_index_2d]

        outhits = deconvolver.shifted_image[
            max_index_2d[0] - 1 : max_index_2d[0] + 2,
            max_index_2d[1] - 1 : max_index_2d[1] + 2,
        ]

        # sum over the middle col and row and dont double count the middle cell
        # outhits = np.sum(outhits[:, 1]) + np.sum(outhits[1,:]) - outhits[1,1]
        outhits = max_signal

        if scale is None:
            scale = max_signal
        fwhm = deconvolver.calculate_fwhm(direction, max_index_2d, scale=scale)

        # print("Indices of the maximum value (2D):", max_index_2d)

    else:
        max_signal = None
        fwhm = None

    hits_ratio = outhits / hits_len

    return max_signal, fwhm, hits_ratio


# -------- ------- SETUP -------- -------
results_folder = "/home/rileyannereid/workspace/geant4/simulation-results/aperture-collimation/61-2-400/"
data_folder = (
    "/home/rileyannereid/workspace/geant4/simulation-data/aperture-collimation/"
)
maxpixel = 60
pix_int = 4
incs = range(pix_int, maxpixel + pix_int, pix_int)
niter = 20

# -------- ------- get the deconvolved image with the mask to get the FWHM and signal -------- -------
simulate = False
txt = True
hitsonly = False

scale = None
center_hits = None
include_hits_effect = False


for direction in ["0", "x", "y", "xy"]:
    print(direction)
    if direction != "0":
        hits_ratios = []
        for ii, inc in enumerate(incs):
            avg_hits_ratio = 0
            for i in range(niter):
                _, _, hits_ratio = run_geom_corr(
                    inc,
                    direction,
                    i,
                    results_folder,
                    simulate=simulate,
                    txt=txt,
                    hitsonly=hitsonly,
                    scale=1,
                    data_folder=data_folder,
                )
                avg_hits_ratio += hits_ratio
            hits_ratios.append(avg_hits_ratio / niter)
        np.savetxt(
            f"{results_folder}{direction}-hits-ratio.txt",
            np.array(hits_ratios),
            delimiter=", ",
            fmt="%.14f",
        )

    else:
        avg_hits_ratio = 0
        for i in range(niter):
            _, _, hits_ratio = run_geom_corr(
                0,
                direction,
                i,
                results_folder,
                simulate=simulate,
                txt=txt,
                hitsonly=hitsonly,
                scale=1,
                data_folder=data_folder,
            )
            avg_hits_ratio += hits_ratio

        np.savetxt(
            f"{results_folder}{direction}-hits-ratio.txt",
            np.array([avg_hits_ratio / niter]),
            delimiter=", ",
            fmt="%.14f",
        )
