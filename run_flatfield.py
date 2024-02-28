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
    det_size_cm = 4.956  # cm
    pixel = 0.28 * 3 # mm
    pixel_cm = pixel * 0.1

    # ---------- coded aperture set up ---------
    # set number of elements
    n_elements_original = 59
    multiplier = 1

    # focal length
    distance = 3.47  # cm

    #det_size_cm = 2.82 #4.984  # cm
    #pixel = 0.2 #0.186666667  # mm
    #pixel_cm = pixel * 0.1  # cm

    # ---------- coded aperture set up ---------

    # set number of elements
    #n_elements_original = 47 #89
    #multiplier = 3

    element_size = pixel * multiplier
    n_elements = (2 * n_elements_original) - 1

    mask_size = element_size * n_elements
    # no trim needed for custom design
    trim = None
    mosaic = True

    # thickness of mask
    det_thickness = 300  # um
    thickness = 300  # um

    # focal length
    #distance = 2.2 # 4.49  # cm

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
        n_particles = 1e5
    else:
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
    energy_level = 500  # keV

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
    #print(np.sum(np.loadtxt(fname)))

    if simulate and not hitsonly:
        simulation_engine.run_simulation(fname, build=False, rename=True)

        # get the raw hits
        myhits = Hits(fname=fname, experiment=False)
        myhits.get_det_hits(
            remove_secondaries=False, second_axis="y", energy_level=energy_level
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
            delta_decoding=False,
            rotate=True
        )
        dc_txt = results_save + "_dc.txt"
        np.savetxt(dc_txt,deconvolver.deconvolved_image)
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
            plot_deconvolved_heatmap=True,
            plot_raw_heatmap=True,
            save_raw_heatmap=results_save + "_raw.png",
            save_deconvolve_heatmap=results_save + "_dc.png",
            plot_signal_peak=False,
            plot_conditions=False,
            hits_txt=True,
            delta_decoding=False,
            rotate=True
        )
        dc_txt = results_save + "_dc.txt"
        np.savetxt(dc_txt,deconvolver.deconvolved_image)
    else:
        # only getting raw hits
        if simulate:
            simulation_engine.run_simulation(fname, build=False, rename=True)
            myhits = Hits(fname=fname, experiment=False)
            myhits.get_det_hits(
                remove_secondaries=True, second_axis="y", energy_level=energy_level
            )
            hits_len = len(myhits.hits_dict["Position"])
        if txt:
            myhits = Hits(fname=fname, experiment=False, txt_file=True)
            hits_len = np.sum(np.loadtxt(fname))
        else:
            print('RE PROCESSING CSV')
            myhits = Hits(fname=fname, experiment=False)
            myhits.get_det_hits(
                remove_secondaries=False, second_axis="y", energy_level=energy_level
            )
            hits_len = len(myhits.hits_dict["Position"])
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
                delta_decoding=False,
                rotate=True
            )
            dc_txt = results_save + "_dc.txt"
            np.savetxt(dc_txt,deconvolver.deconvolved_image)
    
    if not hitsonly:
        # find the max index
        max_index_flat = np.argmax(deconvolver.deconvolved_image)
        max_index_2d = np.unravel_index(
            max_index_flat, deconvolver.deconvolved_image.shape
        )

        # shift up to the noise floor so that the noise floor is at 0
        #deconvolver.shift_noise_floor_ptsrc(max_index_2d[0], max_index_2d[1])
        max_signal = deconvolver.deconvolved_image[max_index_2d]

        if scale is None:
            scale = max_signal
        fwhm = deconvolver.calculate_fwhm(direction, max_index_2d, scale=scale)
        print(results_save)

        # print("Indices of the maximum value (2D):", max_index_2d)
    else:
        max_signal = None
        fwhm = None
    
    #max_signal = None
    #fwhm = None
    return max_signal, fwhm, hits_len


# -------- ------- SETUP -------- -------
data_folder = "/home/rileyannereid/workspace/geant4/simulation-data/59-fwhm/"
results_folder = "/home/rileyannereid/workspace/geant4/simulation-results/59-fwhm/"

maxpixel = 59//2 #71 #134
pix_int = 2 #8
incs = range(pix_int, maxpixel, pix_int)
niter = 8

# -------- -------STEP 1 : need to get the raw hits (shielding stays, but no mask) -------- -------
# need to comment out the physical mask vols in detector construction
step1 = False
simulate = True
txt = False
hitsonly = True

if step1:
    for direction in ["0","xy"]:
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
                )
                avg_hits += nhits
            center_hits = avg_hits / niter
            np.savetxt(
                f"{results_folder}{direction}-hits.txt",
                np.array([center_hits]),
                delimiter=", ",
                fmt="%.14f",
            )

# -------- ------- STEP 2 : get the deconvolved image with the mask to get the FWHM and signal -------- -------
step2 = True
simulate = False
txt = True
hitsonly = False

scale = 1
center_hits = None
include_hits_effect = True

if step2:
    for direction in ["0","xy"]:
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

"""
results_folder = (
    "/home/rileyannereid/workspace/geant4/simulation-results/fwhm-figure/47-2-15/"
)
data_folder = "/home/rileyannereid/workspace/geant4/simulation-data/47-2-15/"

results_folder = "/home/rileyannereid/workspace/geant4/simulation-results/fwhm-figure/47-2-15/"

simulate=False
txt=True
hitsonly=False
#incs = [0,3,60,63]
#incs = [60,63]
incs = [0]
# run just one or two spots
for inc in incs:
    signal, fwhm, _ = run_geom_corr(
        inc,
        "xy",
        0,
        results_folder,
        simulate=simulate,
        txt=txt,
        hitsonly=hitsonly,
        scale=None,
        data_folder=data_folder,
    )

rd = "/home/rileyannereid/workspace/geant4/simulation-results/"
center_txt = np.loadtxt(rd + "47-2-300/47-2.2-0-xy-0_1.00E+06_Mono_600_raw.txt")
center_txt_3 = np.loadtxt(rd + "47-2-300/47-2.2-3-xy-0_1.00E+06_Mono_600_raw.txt")

edge_txt = np.loadtxt(rd + "47-2-300/47-2.2-60-xy-0_1.00E+06_Mono_600_raw.txt")
edge_txt_63 = np.loadtxt(rd + "47-2-300/47-2.2-63-xy-0_1.00E+06_Mono_600_raw.txt")

edge_txt_thick = np.loadtxt(rd + "47-2-15/47-2.2-60-xy-0_1.00E+06_Mono_6000_raw.txt")
edge_txt_thick_63 = np.loadtxt(rd + "47-2-15/47-2.2-63-xy-0_1.00E+06_Mono_6000_raw.txt")

results_dir = "/home/rileyannereid/workspace/geant4/simulation-results/fwhm-figure/"

thin_raw_center = results_dir + "47-2-300/47-2.2-c-xy-0_1.00E+06_Mono_600_raw.txt"
np.savetxt(thin_raw_center, center_txt+center_txt_3)

thick_raw_edge = results_dir + "47-2-15/47-2.2-e-xy-0_1.00E+06_Mono_6000_raw.txt"
thin_raw_edge = results_dir + "47-2-300/47-2.2-e-xy-0_1.00E+06_Mono_600_raw.txt"

np.savetxt(thin_raw_edge, edge_txt+edge_txt_63)
np.savetxt(thick_raw_edge, edge_txt_thick+edge_txt_thick_63)
"""