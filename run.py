from simulation_engine import SimulationEngine
from hits import Hits
from scattering import Scattering
from deconvolution import Deconvolution

from macros import find_disp_pos
import numpy as np

# construct = CA and TD
# source = DS and PS

simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=True)

# timepix design
det_size_cm = 1.408
pixel = 0.055  # mm

# ---------- coded aperture set up ---------

# set number of elements
n_elements_original = 11
multiplier = 22

element_size = pixel * multiplier
n_elements = (2 * n_elements_original) - 1

mask_size = element_size * n_elements
# set edge trim - can't use all pixels to downsample to integer amount
trim = 7
mosaic = True
thickness = 500  # um
distance = 2  # cm

n_particles = 1e8
# -------------------------------------

# -------- pinhole set up -------------
"""
rank = 1
element_size = 1.76/2 # mm
pixels_downsample = 16 #int(10 * det_size_cm / (2 * element_size))
n_elements = rank
mask_size = det_size_cm * 10 # convert to mm
trim = None
mosaic = False

# -------------------------------------

# thickness of mask
thickness = 500  # um

# focal length
distance = 2.4  # cm

# number of particles to simulate
n_particles = 1e9
"""
# --------------set up simulation---------------
signals = []
for i in range(3):
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
    )

    # --------------set up source---------------
    energy_type = "Mono"
    energy_level = 500  # keV

    # --------------set up data naming---------------
    fname_tag = f"sphere-no-aperture-{i}"
    fname = f"../simulation-data/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}.csv"

    simulation_engine.set_macro(
        n_particles=n_particles,
        energy_keV=[energy_type, energy_level, None],
        sphere=True,
        radius_cm=4,
        progress_mod=int(n_particles / 10),  # set with 10 steps
        fname_tag=fname_tag,
    )

    # --------------RUN---------------
    simulation_engine.run_simulation(fname, build=True)

    # ---------- process results -----------
    myhits = Hits(fname=fname, experiment=False)
    myhits.get_det_hits()

    # deconvolution steps
    deconvolver = Deconvolution(myhits, simulation_engine)

    _, _, heatmap, _ = deconvolver.deconvolve(
        plot_deconvolved_heatmap=False,
        plot_raw_heatmap=True,
        save_raw_heatmap=f"../simulation-results/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_raw.png",
        save_deconvolve_heatmap=f"../simulation-results/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_dc.png",
        downsample=1,
        trim=trim,
        plot_signal_peak=True,
        plot_conditions=False,
        save_peak=f"../simulation-results/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_peak.png",
    )

    signal = np.sum(heatmap, axis=1)

    signals.append(signal)

# total counts
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)

for signal in signals:
    ax.plot(signal)
    avg = np.average(signal)
    noise = np.sqrt(signal)
    avg_noise = np.average(noise)

    print(avg, avg_noise)
    ax.hlines(y=avg, xmin=0, xmax=242, linewidth=2, color="r")
    plt.hlines(y=avg - noise, xmin=0, xmax=242, linewidth=2, color="r")
    plt.hlines(y=avg + noise, xmin=0, xmax=242, linewidth=2, color="r")

plt.savefig("no-aperture-no-cap.png")
