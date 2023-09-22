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


def run_flat(fudge, inc, dir, i, cone=False, vmax=None):
    fudge = fudge[0]
    simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=True)

    # general detector design
    det_size_cm = 3.05  # cm
    pixel = 0.5  # mm

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
    thickness = 400  # um

    # focal length
    distance = 2  # cm

    fake_radius = 1

    # set distance of the source to the detector
    src_plane_distance = (1111 * 0.45) + 500
    if dir == "xy":
        theta = np.arctan(
            (pixel * inc * fudge * np.sqrt(2)) * 0.1 / distance
        )  # np.sqrt(2) - add sqrt(2) for diagonal
    else:
        theta = np.arctan((pixel * inc * fudge) * 0.1 / distance)
    src_pos_y = np.tan(theta) * src_plane_distance
    new_x = src_pos_y * np.cos(np.deg2rad(45))
    new_y = src_pos_y * np.sin(np.deg2rad(45))

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
    formatted_theta = "{:.0f}p{:03d}".format(int(theta), int((theta % 1) * 100))
    if dir != "0":
        fname_tag = (
            f"{n_elements_original}-{distance}-{formatted_theta}-{dir}-f-{i}-ptsrc"
        )
    else:
        fname_tag = f"{n_elements_original}-{distance}-{formatted_theta}-f-{i}-ptsrc"
    fname = f"../simulation-data/aperture-collimation/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}.csv"

    # set angular extent of the cone pointing on axis
    maxtheta = np.rad2deg(
        np.arctan((det_size_cm * np.sqrt(2) / 2) / src_plane_distance)
    )

    if dir == "xy":
        newpos = np.sqrt(new_x**2 + new_y**2)
        src_det_dist = np.sqrt(newpos**2 + src_plane_distance**2)
        alpha = np.arctan((pixel * inc * fudge) * 0.1 / distance)
    else:
        src_det_dist = np.sqrt(src_pos_y**2 + src_plane_distance**2)
        alpha = theta

    omega = maxtheta
    p1 = (
        src_det_dist
        * np.tan(np.deg2rad(omega))
        * np.cos(np.deg2rad(omega))
        / np.cos(np.deg2rad(alpha + omega))
    )
    p2 = (
        src_det_dist
        * np.tan(np.deg2rad(omega))
        * np.cos(np.deg2rad(omega))
        / np.cos(np.deg2rad(alpha - omega))
    )

    a = (p1 + p2) / 2
    b = src_det_dist * np.tan(np.deg2rad(omega)) * a / (np.sqrt(p1 * p2))

    # calculate the cone area
    cone_area = np.pi * a * b
    if dir == "xy":
        simulation_engine.set_macro(
            n_particles=int(n_particles),
            energy_keV=[energy_level],
            surface=False,
            positions=[[new_x, new_y, -500]],
            directions=[1],
            progress_mod=int(n_particles / 10),  # set with 10 steps
            fname_tag=fname_tag,
            detector_dim=det_size_cm,
        )
    elif dir == "x":
        simulation_engine.set_macro(
            n_particles=int(n_particles),
            energy_keV=[energy_level],
            surface=False,
            positions=[[src_pos_y, 0, -500]],
            directions=[1],
            progress_mod=int(n_particles / 10),  # set with 10 steps
            fname_tag=fname_tag,
            detector_dim=det_size_cm,
        )
    elif dir == "y":
        simulation_engine.set_macro(
            n_particles=int(n_particles),
            energy_keV=[energy_level],
            surface=False,
            positions=[[0, src_pos_y, -500]],
            directions=[1],
            progress_mod=int(n_particles / 10),  # set with 10 steps
            fname_tag=fname_tag,
            detector_dim=det_size_cm,
        )
    else:
        simulation_engine.set_macro(
            n_particles=int(n_particles),
            energy_keV=[energy_level],
            surface=False,
            positions=[[0, 0, -500]],
            directions=[1],
            progress_mod=int(n_particles / 10),  # set with 10 steps
            fname_tag=fname_tag,
            detector_dim=det_size_cm,
        )
    # --------------RUN---------------
    simulation_engine.run_simulation(fname, build=False, rename=True)

    # ---------- process results -----------

    myhits = Hits(fname=fname, experiment=False)
    myhits.get_det_hits(
        remove_secondaries=True, second_axis="y", energy_level=energy_level
    )

    # deconvolution steps
    deconvolver = Deconvolution(myhits, simulation_engine)

    # directory to save results in
    results_dir = "../simulation-results/aperture-collimation/with-correction/"
    results_tag = f"{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}"
    results_save = results_dir + results_tag

    deconvolver.deconvolve(
        downsample=n_elements_original,
        trim=trim,
        vmax=vmax,
        plot_deconvolved_heatmap=True,
        plot_raw_heatmap=True,
        save_raw_heatmap=results_save + "_raw.png",
        save_deconvolve_heatmap=results_save + "_dc.png",
        plot_signal_peak=False,
        plot_conditions=False,
    )
    inds = (32 + 2 * ((inc / 2) - 1), 28 - 2 * ((inc / 2) - 1))
    print("checked index #", inds)

    if cone:
        if dir == "x":
            return deconvolver.deconvolved_image[int(inds[0]), 30], cone_area
        elif dir == "y":
            return deconvolver.deconvolved_image[30, int(inds[1])], cone_area
        elif dir == "xy":
            return deconvolver.deconvolved_image[int(inds[0]), int(inds[1])], cone_area
        else:
            return deconvolver.deconvolved_image[30, 30], cone_area
    else:
        if dir == "x":
            return deconvolver.deconvolved_image[int(inds[0]), 30]
        elif dir == "y":
            return deconvolver.deconvolved_image[30, int(inds[1])]
        elif dir == "xy":
            return deconvolver.deconvolved_image[int(inds[0]), int(inds[1])]
        else:
            return deconvolver.deconvolved_image[30, 30]


# --- --- --- --- optimization to get the pixel correct --- --- --- ---
dec, center_cone = run_flat([0.99], 28, "x", 0, cone=True, vmax=None)
"""
initial_guess = 0.99
bounds = [(0.97, 1.02)]

maxpixel = 30
pix_int = 2
incs = range(pix_int, maxpixel + pix_int, pix_int)
data_folder = "/home/rileyannereid/workspace/geant4/simulation-results/aperture-collimation/with-correction/"

optimization_file = "-f.txt"
final_opts = []
for dir in ["x", "y", "xy"]:
    optimizations = []
    # Use the minimize function with the negative objective to find the maximum
    for inc in incs:
        result = minimize(
            lambda x: -run_flat(x, inc, dir, 0),
            initial_guess,
            bounds=bounds,
            options={"maxiter": 15},
        )

        # Print the optimization result
        if result.success:
            print("Optimal solution found:")
            print("x =", result.x)
            print("Maximum value =", -result.fun)
        else:
            print("Optimization failed:", result.message)

        optimizations.append(result.x)
    optimizations_array = np.array(optimizations)
    np.savetxt(
        f"{data_folder}{dir}{optimization_file}",
        optimizations_array,
        delimiter=", ",
        fmt="%.14f",
    )
    final_opts.append(optimizations)

# --- --- --- --- now run 3 at each to get average signal and size of cone --- --- --- ---

signal_strengths = []
for i in range(3):
    dec, center_cone = run_flat([1], 0, "0", i, cone=True, vmax=None)
    signal_strengths.append(dec)

signal_norm = np.mean(np.array(signal_strengths))
cone_norm = center_cone

all_cones = []
all_signals = []
for dir, ff in zip(dirs, final_opts):
    cones = []
    signals = []
    for inc, f in zip(incs, ff):
        signal_avg = 0
        for i in range(3):
            dec, cone = run_flat([f], inc, dir, i, cone=True, vmax=signal_norm)
            signal_avg += dec
        # save the average for the pixel and the cone size
        signals.append(signal_avg / 3)
        cones.append(cone)
    np.savetxt(
        f"{data_folder}{dir}-cone.txt", np.array(cones), delimiter=", ", fmt="%.14f"
    )
    np.savetxt(
        f"{data_folder}{dir}-signal.txt", np.array(signals), delimiter=", ", fmt="%.14f"
    )
    all_cones.append(cones)
    all_signals.append(signals)

# --- --- --- --- get the final normalization --- --- --- ---

x_signals_avg = [
    (all_cones[0][i] / cone_norm) * xs / signal_norm
    for i, xs in enumerate(all_signals[0])
]
y_signals_avg = [
    (all_cones[1][i] / cone_norm) * ys / signal_norm
    for i, ys in enumerate(all_signals[1])
]
xy_signals_avg = [
    (all_cones[2][i] / cone_norm) * xys / signal_norm
    for i, xys in enumerate(all_signals[2])
]

# save the final data
data_2dx = np.array(x_signals_avg)
data_2dy = np.array(y_signals_avg)
data_2dxy = np.array(xy_signals_avg)

# Define the output file path
output_filex = "x-signals-norm.txt"
output_filey = "y-signals-norm.txt"
output_filexy = "xy-signals-norm.txt"

# Save the 2D array to the text file
np.savetxt(f"{data_folder}{output_filex}", data_2dx, delimiter=", ", fmt="%.14f")
np.savetxt(f"{data_folder}{output_filey}", data_2dy, delimiter=", ", fmt="%.14f")
np.savetxt(f"{data_folder}{output_filexy}", data_2dxy, delimiter=", ", fmt="%.14f")

"""
"""

# save data here just in case
# x
fx = [
    0.98997616,
    0.98999416,
    0.99,
    0.98999924,
    0.99000025,
    0.99,
    0.98999993,
    0.99000094,
    0.99,
    0.98999835,
    0.99001095,
    0.98999702,
    0.99,
    0.99000219,
    1.01,
]
# y
fy = [
    0.99000002,
    0.99000079,
    0.99000412,
    0.99002364,
    0.99,
    0.99001873,
    0.98999748,
    0.98998595,
    0.98999644,
    0.99,
    0.99006229,
    0.98998045,
    0.99,
    0.99000011,
    1.01,
]

# diagonal
fd = [
    0.99,
    0.99000611,
    0.99003122,
    0.99,
    0.98998818,
    0.98999197,
    0.98999184,
    0.9899989,
    0.98999992,
    0.99000783,
    0.99000049,
    0.98999899,
    0.98999891,
    0.98999995,
    1.01,
]

x_signals = [
    [289116.00000000006, 289358.00000000006, 289460.00000000006],
    [284610.00000000006, 285682.00000000006, 285732.00000000006],
    [280024.0000000001, 279634.00000000006, 278946.00000000006],
    [271602.0000000001, 271114.0000000001, 271292.00000000006],
    [264224.0, 263176.00000000006, 263554.0000000001],
    [254808.00000000012, 254506.0000000001, 254746.00000000006],
    [244302.0000000001, 244946.0000000001, 244774.0000000001],
    [234392.00000000003, 233536.00000000006, 235416.00000000006],
    [225632.00000000006, 225282.00000000006, 225856.00000000003],
    [215864.00000000006, 215272.00000000006, 216060.00000000006],
    [206312.00000000006, 207862.00000000006, 207588.00000000006],
    [197122.00000000003, 197057.99999999997, 196860.00000000006],
    [187424.00000000003, 186920.00000000006, 186916.00000000003],
    [178828.00000000003, 178388.00000000003, 178124.00000000006],
    [107130.00000000003, 106680.00000000004, 106810.00000000001],
]

y_signals = [
    [290196.00000000006, 289078.00000000006, 289134.0000000001],
    [286094.00000000006, 285752.00000000006, 285774.00000000006],
    [279972.00000000006, 280502.00000000006, 280132.00000000006],
    [272890.0000000001, 272826.0000000001, 272858.00000000006],
    [265050.00000000006, 263716.0000000001, 265190.00000000006],
    [256032.0000000001, 255692.0000000001, 256608.0000000001],
    [249042.00000000012, 249542.0000000001, 250552.0000000001],
    [236664.00000000003, 236490.00000000006, 237048.0000000001],
    [227654.00000000006, 228698.00000000006, 227798.00000000006],
    [218090.00000000006, 219030.00000000006, 218706.00000000003],
    [214410.00000000006, 215014.00000000006, 214290.00000000003],
    [201162.00000000006, 201208.00000000006, 202020.00000000006],
    [191968.00000000006, 190450.00000000006, 191778.00000000006],
    [188834.00000000006, 189064.00000000003, 189310.00000000006],
    [112200.0, 113010.0, 112206.00000000001],
]
xy_signals = [
    [287428.00000000006, 287282.0000000001, 286840.0000000001],
    [281876.00000000006, 280964.00000000006, 281288.00000000006],
    [270366.00000000006, 270110.00000000006, 269608.0000000001],
    [254594.00000000003, 254126.0000000001, 254658.0000000001],
    [240024.0000000001, 239494.0000000001, 239386.0000000001],
    [224836.00000000003, 223214.00000000015, 223922.00000000006],
    [210392.00000000006, 209314.00000000006, 209876.00000000006],
    [192120.00000000006, 191354.00000000003, 191748.0],
    [177412.00000000003, 177176.00000000006, 177030.00000000003],
    [164240.00000000006, 164140.00000000006, 163884.00000000003],
    [154560.00000000006, 153806.00000000003, 154070.0],
    [138372.00000000006, 138682.00000000003, 138352.00000000006],
    [126500.00000000004, 125916.00000000004, 126512.00000000004],
    [120728.00000000003, 120428.00000000003, 120982.00000000001],
    [46852.00000000001, 45992.00000000001, 46000.0],
]

"""
