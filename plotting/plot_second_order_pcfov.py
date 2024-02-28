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


n_particles_frac = np.logspace(0, -2, 8)
errors = []
for npf in n_particles_frac[1:]:

    n_particles = (
        int(
            (8e7 * (5.030 * 2) ** 2) * (np.cos(np.deg2rad(45)) - np.cos(np.deg2rad(70)))
        )
        * npf
    )

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
    # signal = np.loadtxt("../simulation-results/rings/59-3.47-35p00-deg_1.00E+09_Mono_100_dc.txt")
    cpfov = pcfov + pcfov_rotated

    combined = np.loadtxt(
        "../simulation-results/rings/59-3.47-22-45-deg-fov_1.53E+09_Mono_100_combined.txt"
    )

    pixel_count = 59
    center_pixel = int(59 / 2)
    geometric_factor = 18
    pcfov_signal = 0
    fcfov_signal = 0
    signal_count = 0
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

            if np.rad2deg(angle) < (22 + 0.5):
                signal_count += 1
                fcfov_signal += signal[y, x]
                pcfov_signal += signal[y, x] + pcfov[y, x]

    px_factor = signal_count / (pixel_count**2)

    print("recorded flux", fcfov_signal / (geometric_factor * px_factor))
    print("new flux", pcfov_signal / (geometric_factor * px_factor))
    error = 100 - (
        100
        * (
            (pcfov_signal / (geometric_factor * px_factor))
            / (fcfov_signal / (geometric_factor * px_factor))
        )
    )

    print(error)
    errors.append(error)

color = "#39329E"
fig, ax1 = plt.subplots(figsize=(5.7, 2))

ax1.set_axisbelow(True)
ax1.grid(True, linestyle="--", color="lightgrey", linewidth=0.5)
ax1.plot(
    0.5 * n_particles_frac[1:] / n_particles_frac[1],
    np.abs(errors),
    color=color,
    linewidth=2,
)
ax1.set_xscale("log")
ax1.set_ylabel(r"% Error", fontsize=8)
ax1.set_xlabel("Fraction of intensity of FCFOV source", fontsize=8, color="black")
ax1.tick_params(axis="both", labelsize=8)
ax1.xaxis.labelpad = 0.2
ax1.yaxis.labelpad = 0.2
plt.savefig(
    "../simulation-results/final-images/pcofv_artifacts.png",
    dpi=500,
    transparent=False,
    bbox_inches="tight",
    pad_inches=0.02,
)
