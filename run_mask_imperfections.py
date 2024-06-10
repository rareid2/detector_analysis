from simulation_engine import SimulationEngine
from hits import Hits
from scattering import Scattering
from deconvolution import Deconvolution
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from macros import find_disp_pos
from scipy.optimize import minimize
import numpy as np
import os
import copy

simulate = False
txt = True

# -------- ------- SETUP -------- -------
data_folder = "/home/rileyannereid/workspace/geant4/simulation-data/mask/"
results_folder = "/home/rileyannereid/workspace/geant4/simulation-results/mask/"

nthreads = 14

if simulate:
    simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=True)
else:
    simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=False)

# general detector design
det_size_cm = 1.331  # cm
pixel = 0.403333333  # mm
pixel_cm = pixel * 0.1

# ---------- coded aperture set up ---------
# set number of elements
n_elements_original = 11
multiplier = 3

# focal length
distance = 2  # cm

element_size = pixel * multiplier
n_elements = (2 * n_elements_original) - 1

mask_size = element_size * n_elements
# no trim needed for custom design
trim = None
mosaic = True

# thickness of mask
det_thickness = 300  # um
thickness = 500  # um

fake_radius = 1

# set distance of the source to the detector
world_offset = 1111 * 0.45
detector_loc = world_offset - ((det_thickness / 2) * 1e-4)
ca_pos = world_offset - distance - (((thickness / 2) + (det_thickness / 2)) * 1e-4)
src_plane_distance = world_offset + 500

# pixel location
ind = 0
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
n_particles = 1e7
fwhms = []
maxes = []
avg_snr = []
for i in range(5):
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
    energy_level = 20  # keV

    # --------------set up data naming---------------
    fname_tag = f"{n_elements_original}-{distance}_{n_particles:.2E}_{energy_type}_{energy_level}_CAD-extend-{i}"
    if txt:
        fname = f"{results_folder}{fname_tag}_raw.txt"
    else:
        fname = f"{data_folder}{fname_tag}.csv"
    """
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
    """
    simulation_engine.set_macro(
        n_particles=int(n_particles),
        energy_keV=[energy_type, energy_level, None],
        surface=True,
        progress_mod=int(n_particles / 10),  # set with 10 steps
        fname_tag=fname_tag,
        confine=False,
        detector_dim=det_size_cm,
        theta=10,
        theta_lower=0,
        ring=True,
        # radius_cm=3,
    )

    # --------------RUN AND PROCESS---------------
    results_save = f"{results_folder}{fname_tag}"

    if not txt:
        if simulate:
            simulation_engine.run_simulation(fname, build=True, rename=True)

        # get the raw hits
        for hi in range(nthreads):
            print(hi)
            fname_hits = fname[:-4] + "-{}.csv".format(hi)
            myhits = Hits(fname=fname_hits, experiment=False, txt_file=txt)
            hits_dict, sec_brehm, sec_e = myhits.get_det_hits(
                remove_secondaries=False, second_axis="y"
            )
            if hi != 0:
                # update fields in hits dict
                myhits.hits_dict["Position"].extend(hits_copy.hits_dict["Position"])
                myhits.hits_dict["Energy"].extend(hits_copy.hits_dict["Energy"])

                hits_copy = copy.copy(myhits)
            else:
                hits_copy = copy.copy(myhits)
            hits_len = len(myhits.hits_dict["Position"])
    elif txt:
        raw = np.loadtxt(f"{results_save}_raw.txt")
        raw = np.flipud(raw)
        np.savetxt(f"{results_save}_raw.txt", raw)
        myhits = Hits(fname=fname, experiment=False, txt_file=True)
    deconvolver = Deconvolution(myhits, simulation_engine)
    deconvolver.deconvolve(
        downsample=int(multiplier * n_elements_original),
        trim=trim,
        plot_deconvolved_heatmap=True,
        plot_raw_heatmap=True,
        save_raw_heatmap=results_save + "_raw.png",
        save_deconvolve_heatmap=results_save + "_dc.png",
        plot_signal_peak=False,
        plot_conditions=False,
        hits_txt=txt,
        delta_decoding=False,
        rotate=True,
    )
    np.savetxt(results_save + "_dc.txt", deconvolver.deconvolved_image)
    print(np.amax(deconvolver.deconvolved_image))
    max_index_flat = np.argmax(deconvolver.deconvolved_image)
    max_index_2d = np.unravel_index(max_index_flat, deconvolver.deconvolved_image.shape)

    # shift up to the noise floor so that the noise floor is at 0
    # deconvolver.shift_noise_floor_ptsrc(max_index_2d[0], max_index_2d[1])
    max_signal = deconvolver.deconvolved_image[max_index_2d]

    scale = max_signal
    # fwhm = deconvolver.calculate_fwhm("0", max_index_2d, scale=scale)
    maxes.append(np.amax(deconvolver.deconvolved_image))
    # fwhms.append(fwhm)
    # print(fwhm)
    pixel_count = multiplier * n_elements_original
    center_pixel = pixel_count // 2
    pixel_size = pixel * 0.1
    theta_bound = 2.5
    signal_sum = 0
    signal = deconvolver.deconvolved_image
    signals_snr = []
    for x in range(pixel_count):
        for y in range(pixel_count):
            relative_x = (x - center_pixel) * pixel_size
            relative_y = (y - center_pixel) * pixel_size

            aa = np.sqrt(relative_x**2 + relative_y**2)

            # find the geometrical theta angle of the pixel
            angle = np.arctan(aa / distance)
            angle = np.rad2deg(angle)

            if 0 <= angle <= theta_bound:
                signal_sum += signal[y, x]
                signals_snr.append(signal[y, x])

                rect = patches.Rectangle(
                    (x - 0.5, y - 0.5),
                    1,
                    1,
                    linewidth=0.5,
                    edgecolor="black",
                    facecolor="none",
                )

                plt.gca().add_patch(rect)
    plt.imshow(signal)
    plt.savefig(f"{results_save}_test.png")
    rho = 0.5
    t = 0
    xi = 0
    snrs = []
    nt = 11**2
    for ss in signals_snr:
        psi = ss / signal_sum
        snr_mura = (
            np.sqrt(nt * signal_sum)
            * np.sqrt(rho * (1 - rho))
            * (1 - t)
            * psi
            / np.sqrt((1 - t) * (rho + (1 - 2 * rho) * psi) + t + xi)
        )
        snrs.append(snr_mura)

    print("avg", np.average(np.array(snrs)))
    avg_snr.append(np.average(np.array(snrs)))
print("average average", np.average(np.array(avg_snr)))
