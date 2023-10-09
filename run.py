from simulation_engine import SimulationEngine
from hits import Hits
from scattering import Scattering
from deconvolution import Deconvolution

from macros import find_disp_pos
import numpy as np
import os
import copy

# construct = CA and TD
# source = DS and PS

simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=True)

det_size_cm = 3.05
pixel = 0.25  # mm
pixel_cm = pixel * 0.1

# ---------- coded aperture set up ---------

# set number of elements
n_elements_original = 61
multiplier = 2

element_size = pixel * multiplier
n_elements = (2 * n_elements_original) - 1

mask_size = element_size * n_elements
# set edge trim - can't use all pixels to downsample to integer amount
trim = None
mosaic = True

# -------------------------------------

# -------- pinhole set up -------------
"""
rank = 1
element_size = 1.76 / 8  # mm
n_elements = rank
mask_size = det_size_cm * 10  # convert to mm
trim = None
mosaic = False
"""
# -------------------------------------

# thickness of mask
thickness = 400  # um

# focal length
distance = 2  # cm

det_thickness = 300  # um

# ------------------- simulation parameters ------------------
n_particles = 1e6
axis = 0
theta = None
fake_radius = 1

# set distance of the source to the detector
world_offset = 1111 * 0.45
detector_loc = world_offset - ((det_thickness / 2) * 1e-4)
ca_pos = world_offset - distance - (((thickness / 2) + (det_thickness / 2)) * 1e-4)
src_plane_distance = world_offset + 500

# pixel location
pix_ind = 2
direction = "xy"
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
for i in [0, 2]:
    # --------------set up data naming---------------
    fname_tag = f"{n_elements_original}-{i}-res"
    fname = f"../simulation-data/aperture-collimation/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}.csv"

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

    # --------------RUN---------------
    # simulation_engine.run_simulation(fname, build=False, rename=True)

    # ---------- process results -----------
    myhits = Hits(fname=fname, experiment=False)
    myhits.get_det_hits(
        remove_secondaries=True, second_axis="y", energy_level=energy_level
    )
    # myhits.exclude_pcfov(det_size_cm, mask_size/10, distance, radius_cm)

    if i != 0:
        # update fields in hits dict
        myhits.hits_dict["Position"].extend(hits_copy.hits_dict["Position"])
        myhits.hits_dict["Energy"].extend(hits_copy.hits_dict["Energy"])
        myhits.hits_dict["Vertices"].extend(hits_copy.hits_dict["Vertices"])

        hits_copy = copy.copy(myhits)
    else:
        hits_copy = copy.copy(myhits)

    print(fname_tag)

    # deconvolution steps
    deconvolver = Deconvolution(myhits, simulation_engine)

# directory to save results in
results_dir = "../simulation-results/aperture-collimation/"
results_tag = f"{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}"
results_save = results_dir + results_tag

deconvolver.deconvolve(
    downsample=int(multiplier * n_elements_original),
    trim=trim,
    vmax=None,
    plot_deconvolved_heatmap=True,
    plot_raw_heatmap=True,
    save_raw_heatmap=results_save + "_raw.png",
    save_deconvolve_heatmap=results_save + "_dc.png",
    plot_signal_peak=False,
    plot_conditions=False,
    save_peak=results_save + "_peak.png",
    normalize_signal=False,
    axis=axis,
)
deconvolver.plot_3D_signal(results_save + "_2Dpeak.png")
