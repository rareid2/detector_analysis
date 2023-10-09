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

# need to run for each indices in each direction, 3 times


def run_geom_corr(ind, direction, i, vmax=None):
    simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=True)

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
    fname_tag = f"{n_elements_original}-{distance}-{ind}-{direction}-{i}"
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
    if i < 5:
        pass
    elif direction == "0":
        pass
    else:
        simulation_engine.run_simulation(fname, build=False, rename=True)

    # ---------- process results -----------

    myhits = Hits(fname=fname, experiment=False)
    myhits.get_det_hits(
        remove_secondaries=True, second_axis="y", energy_level=energy_level
    )
    # print(len(myhits.hits_dict["Position"]))
    # deconvolution steps
    deconvolver = Deconvolution(myhits, simulation_engine)

    # directory to save results in
    results_dir = "../simulation-results/aperture-collimation/61-2-400-d3-2p25/"
    results_tag = f"{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}"
    results_save = results_dir + results_tag

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
    )

    fwhm = deconvolver.calculate_fwhm()

    # return len(myhits.hits_dict["Position"])
    # Find the indices of the maximum value in the flattened array
    max_index_flat = np.argmax(deconvolver.deconvolved_image)

    # Convert the flattened index to 2D indices
    max_index_2d = np.unravel_index(max_index_flat, deconvolver.deconvolved_image.shape)

    # Print the result
    print("Indices of the maximum value (2D):", max_index_2d)
    noise = np.mean(deconvolver.deconvolved_image[5:50, 5:50])

    return deconvolver.deconvolved_image[max_index_2d] - noise, fwhm


# setup
data_folder = "/home/rileyannereid/workspace/geant4/simulation-results/aperture-collimation/61-2-400-d3-2p25/"
maxpixel = 60
pix_int = 4
incs = range(pix_int, maxpixel + pix_int, pix_int)
niter = 20


# RUN GEOMETRY NORMALIZATION ------------------ normalize total number of possible hits on detector
# run center hits first
"""
avg_hits = 0
for i in range(niter):
    nhits = run_geom_corr(0, "0", i)
    avg_hits += nhits

# save it
center_hits = avg_hits / niter
np.savetxt(
    f"{data_folder}center-hits-norm.txt",
    np.array([center_hits]),
    delimiter=", ",
    fmt="%.14f",
)

# run each direction hits - no mask
all_hits = []
for direction in ["x", "y", "xy"]:
    total_hits = []
    for inc in incs:
        avg_hits = 0
        for i in range(niter):
            nhits = run_geom_corr(inc, direction, i)
            avg_hits += nhits
        total_hits.append((avg_hits / niter) / center_hits)
    np.savetxt(
        f"{data_folder}{direction}-hits-norm.txt",
        np.array(total_hits),
        delimiter=", ",
        fmt="%.14f",
    )
    all_hits.append(total_hits)

print(all_hits)
"""
# RUN SIGNAL NORMALIZATION ------------------ normalize total number of possible hits on detector
# run center hits first
"""
avg_signal = 0
fwhms = []
for i in range(niter):
    dec, fwhm = run_geom_corr(0, "0", i)
    avg_signal += dec
    fwhms.append(fwhm)
    np.savetxt(
        f"{data_folder}center-fwhm.txt",
        np.array([fwhms]),
        delimiter=", ",
        fmt="%.14f",
    )

# save it

sig_norm = avg_signal / niter
np.savetxt(
    f"{data_folder}center-sig-norm.txt",
    np.array([sig_norm]),
    delimiter=", ",
    fmt="%.14f",
)

np.savetxt(
    f"{data_folder}center-fwhm.txt",
    np.array([fwhms]),
    delimiter=", ",
    fmt="%.14f",
)

# run each direction signal
all_hits = []
for direction in ["x", "y", "xy"]:
    total_hits = []
    all_fwhms = []
    for inc in incs:
        avg_hits = 0
        fwhms = []
        for i in range(niter):
            nhits, fwhm = run_geom_corr(inc, direction, i)
            avg_hits += nhits
            fwhms.append(fwhm)
        # total_hits.append((avg_hits / niter) / sig_norm)
        all_fwhms.append(fwhms)
        # np.savetxt(
        #    f"{data_folder}{direction}-sig.txt",
        #    np.array(total_hits),
        #    delimiter=", ",
        #    fmt="%.14f",
        # )
        np.savetxt(
            f"{data_folder}{direction}-fwhm.txt",
            np.array(all_fwhms),
            delimiter=", ",
            fmt="%.14f",
        )
    np.savetxt(
        f"{data_folder}{direction}-fwhm.txt",
        np.array(all_fwhms),
        delimiter=", ",
        fmt="%.14f",
    )
    # all_hits.append(total_hits)

# print(all_hits)


# Define the input and output file paths
for direction in ["x", "y", "xy"]:
    input_file_path = f"{data_folder}{direction}-fwhm.txt"
    output_file_path = f"{data_folder}{direction}-average-fwhm.txt"

    # Read the input file
    with open(input_file_path, "r") as input_file:
        lines = input_file.readlines()

    # Initialize an empty list to store the averages
    averages = []

    # Process each row in the input data
    for line in lines:
        # Split the line into a list of numbers, assuming they are separated by commas
        numbers = [float(x) for x in line.strip().split(",")]

        # Calculate the average of the numbers in the row
        row_average = sum(numbers) / len(numbers)

        # Append the average to the list
        averages.append(row_average)

    # Save the averages to the output file
    with open(output_file_path, "w") as output_file:
        for average in averages:
            output_file.write(f"{average:.14f}\n")

    print(f"Averages saved to {output_file_path}")
"""

directions = ["x", "y", "xy"]
for direction in directions:
    file_path = f"{data_folder}{direction}-average-fwhm.txt"  # Change this to the path of your data file
    data = np.loadtxt(file_path)

    import numpy as np
    from scipy.interpolate import CubicSpline

    # 2. Fit a cubic spline curve to the data
    x = np.arange(len(data))  # Generate x values as integers from 0 to N-1
    coefficients = np.polyfit(x, data, 2)  # Fit a first-degree (linear) polynomial
    print(coefficients)
    # The coefficients now contain the slope and intercept of the fitted line.
    a, b, c = coefficients
    print(f"The quadratic polynomial coefficients are: a={a}, b={b}, c={c}")

    # You can create a polynomial function using these coefficients:
    poly_func = np.poly1d(coefficients)
    calculated_values = poly_func(x)

    # 4. Save the calculated values to a new text file
    output_file_path = f"{data_folder}{direction}-fwhm-calc.txt"  # Change this to your desired output file path
    np.savetxt(output_file_path, calculated_values, fmt="%.6f", delimiter="\n")

    print(f"Calculated values saved to {output_file_path}")
