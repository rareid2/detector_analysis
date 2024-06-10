import numpy as np
import cmocean

cmap = cmocean.cm.thermal

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14})


def fmt(x, pos):
    if x == 0:
        return "0"
    else:
        a, b = "{:.1e}".format(x).split("e")
        b = int(b)
        a = float(a)
        if a % 1 > 0.1:
            pass
        else:
            a = int(a)
        return f"{a}"


def fcfov_plane(
    detector_size_mm: float,
    mask_size_mm: float,
    mask_detector_distance_mm: float,
    mask_plane_distance_mm: float,
):
    detector_diagonal_mm = detector_size_mm * np.sqrt(2)
    mask_diagonal_mm = mask_size_mm * np.sqrt(2)

    detector_half_diagonal_mm = detector_diagonal_mm / 2
    mask_half_diagonal_mm = mask_diagonal_mm / 2

    # FCFOV half angle
    theta_fcfov_deg = np.rad2deg(
        np.arctan(
            (mask_half_diagonal_mm - detector_half_diagonal_mm)
            / mask_detector_distance_mm
        )
    )
    # print(theta_fcfov_deg, "half angle")

    # pcfov
    fov = np.rad2deg(
        np.arctan(
            (detector_diagonal_mm + (mask_half_diagonal_mm - detector_half_diagonal_mm))
            / mask_detector_distance_mm
        )
    )
    # print("PCFOV", fov - theta_fcfov_deg, "Half angle pcfov")
    # project this to a distance
    plane_distance_to_detector_mm = mask_detector_distance_mm + mask_plane_distance_mm

    additional_diagonal_mm = (
        np.tan(np.deg2rad(theta_fcfov_deg)) * plane_distance_to_detector_mm
    )

    plane_diagonal_mm = (additional_diagonal_mm + detector_half_diagonal_mm) * 2

    plane_side_length_mm = plane_diagonal_mm / np.sqrt(2)

    # geant asks for half side length

    plane_half_side_length_mm = plane_side_length_mm / 2

    # print(f"FCFOV square plane should be half side length = {plane_half_side_length_mm} mm at a distance {plane_distance_to_detector_mm} mm from detector")

    pcfov = fov - theta_fcfov_deg
    return pcfov, theta_fcfov_deg


n_particles_frac = np.logspace(0, -2, 8)
# npf = n_particles_frac[1]  # used 1 in the paper
errors = []
second_order_errors = []
errors2 = []
second_order_errors2 = []

for npf in n_particles_frac[1:]:

    n_particles = (
        int(
            (8e7 * (5.030 * 2) ** 2) * (np.cos(np.deg2rad(45)) - np.cos(np.deg2rad(70)))
        )
        * npf
    )
    detector_size_mm = 49.56
    mask_size_mm = 98.28
    mask_detector_distance_mm = 34.7
    element_size_mm = 0.84
    mask_plane_distance_mm = 0.15 + 1.0074 + 0.5

    mask_detector_distance_mms = np.linspace(1, 100, 50)

    fcfovs = []
    pcfovs = []
    fovs = []

    for mask_detector_distance_mm in mask_detector_distance_mms:
        pcfov, fcfov = fcfov_plane(
            detector_size_mm,
            mask_size_mm,
            mask_detector_distance_mm,
            mask_plane_distance_mm,
        )
        fcfovs.append(fcfov)
        pcfovs.append(pcfov)
        fovs.append(pcfov / (fcfov / 2))
    fcfov_plane(detector_size_mm, mask_size_mm, 34.7, mask_plane_distance_mm)

    # load data
    pcfov_rotated = np.loadtxt(
        f"../simulation-results/rings/59-3.47-45p00-deg-rotate_{n_particles:.2E}_Mono_100_dc.txt"
    )
    pcfov = np.loadtxt(
        f"../simulation-results/rings/59-3.47-45p00-deg_{n_particles:.2E}_Mono_100_dc.txt"
    )
    signal = np.loadtxt(
        "../simulation-results/rings/59-3.47-22p00-deg_5.90E+08_Mono_100_dc.txt"
    )
    signal2 = np.loadtxt(
        "../simulation-results/rings/59-3.47-35p00-deg_1.00E+09_Mono_100_dc.txt"
    )
    cpfov = pcfov + pcfov_rotated

    combined = np.loadtxt(
        "../simulation-results/rings/59-3.47-22-45-deg-fov_1.53E+09_Mono_100_combined.txt"
    )

    pixel_count = 59
    center_pixel = int(59 / 2)
    geometric_factor = 18

    pcfov_signal = 0
    fcfov_signal = 0
    cpfov_signal = 0
    signal_count = 0

    pcfov_signal2 = 0
    fcfov_signal2 = 0
    cpfov_signal2 = 0
    signal_count2 = 0

    geometric_factor = 18
    distance = 3.47
    pixel = 0.28 * 3  # mm
    pixel_size = pixel * 0.1
    for x in range(pixel_count):
        for y in range(pixel_count):
            relative_x = (x - center_pixel) * pixel_size
            relative_y = (y - center_pixel) * pixel_size

            aa = np.sqrt(relative_x**2 + relative_y**2)

            # find the geometrical theta angle of the pixel
            angle = np.arctan(aa / distance)

            if (22 - 1.4) < np.rad2deg(angle) < (22 + 1.4):
                signal_count += 1
                fcfov_signal += signal[y, x]
                pcfov_signal += signal[y, x] + pcfov[y, x]
                cpfov_signal += signal[y, x] + cpfov[y, x]

            if (35 - 1.4) < np.rad2deg(angle) < (35 + 1.4):
                signal_count2 += 1
                fcfov_signal2 += signal2[y, x]
                pcfov_signal2 += signal2[y, x] + pcfov[y, x]
                cpfov_signal2 += signal2[y, x] + cpfov[y, x]

    px_factor = signal_count / (pixel_count**2)
    px_factor2 = signal_count2 / (pixel_count**2)

    error = 100 - (
        100
        * (
            (pcfov_signal / (geometric_factor * px_factor))
            / (fcfov_signal / (geometric_factor * px_factor))
        )
    )
    second_error = 100 - (
        100
        * (
            (cpfov_signal / (geometric_factor * px_factor))
            / (fcfov_signal / (geometric_factor * px_factor))
        )
    )

    errors.append(np.abs(error))
    second_order_errors.append(np.abs(second_error))

    # larger extent
    error2 = 100 - (
        100
        * (
            (pcfov_signal2 / (geometric_factor * px_factor))
            / (fcfov_signal2 / (geometric_factor * px_factor))
        )
    )
    second_error2 = 100 - (
        100
        * (
            (cpfov_signal2 / (geometric_factor * px_factor2))
            / (fcfov_signal2 / (geometric_factor * px_factor2))
        )
    )
    errors2.append(np.abs(error2))
    second_order_errors2.append(np.abs(second_error2))

line_color = "#EFE34E"
line_color2 = "#F68E42"
fig = plt.figure(figsize=(7, 3))  # 1 row, 2 columns
linewidth = 2
plt.plot(n_particles_frac[1:], errors, linewidth=2, color=line_color)
plt.text(0.01, 65, r"50% extent, second-order included", color=line_color, fontsize=10)

plt.plot(2 * n_particles_frac[1:], second_order_errors, linewidth=2, color=line_color2)
plt.text(0.01, 60, r"50% extent, second-order removed", color=line_color2, fontsize=10)

line_color = "#6B4493"
line_color2 = "#092E59"

# larger extent
plt.plot(n_particles_frac[1:], errors2, linewidth=2, color=line_color)
plt.plot(2 * n_particles_frac[1:], second_order_errors2, linewidth=2, color=line_color2)
plt.text(0.01, 55, r"75% extent, second-order included", color=line_color, fontsize=10)
plt.text(0.01, 50, r"75% extent, second-order removed", color=line_color2, fontsize=10)

plt.xscale("log")
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.xlabel("Relative magnitude of PCFOV source compared to FCFOV source", fontsize=10)
plt.ylabel(
    "% Deviation in Integrated Fluence \n of Outermost Pitch Angle Bin", fontsize=10
)

plt.savefig(
    "../simulation-results/final-images/pcfov_var.png",
    dpi=500,
    bbox_inches="tight",
    pad_inches=0.02,
)
