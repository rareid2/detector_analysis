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
thickness = 100  # um
distance = 2  # cm

n_particles = 1e8
# -------------------------------------

# -------- pinhole set up -------------

rank = 1
element_size = 1.76/4 # mm
n_elements = rank
mask_size = det_size_cm * 10 # convert to mm
trim = None
mosaic = False

# -------------------------------------

# thickness of mask
thickness = 500  # um

# focal length
distance = 1  # cm

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
    fname_tag = f"sphere-pinhole-{i}"
    fname = f"../simulation-data/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}.csv"

    simulation_engine.set_macro(
        n_particles=n_particles,
        energy_keV=[energy_type, energy_level, None],
        sphere=True,
        radius_cm=2.75,
        progress_mod=int(n_particles / 10),  # set with 10 steps
        fname_tag=fname_tag,
    )

    # --------------RUN---------------
    #simulation_engine.run_simulation(fname, build=True)

    # ---------- process results -----------
    myhits = Hits(fname=fname, experiment=False)
    myhits.get_det_hits()

    # deconvolution steps
    deconvolver = Deconvolution(myhits, simulation_engine)

    _, _, heatmap, signal = deconvolver.deconvolve(
        plot_deconvolved_heatmap=False,
        plot_raw_heatmap=True,
        save_raw_heatmap=f"../simulation-results/validating-iso/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_raw.png",
        save_deconvolve_heatmap=f"../simulation-results/validating-iso/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_dc.png",
        downsample=32,
        trim=trim,
        plot_signal_peak=True,
        plot_conditions=False,
        save_peak=f"../simulation-results/validating-iso/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_peak.png",
    )

    signals.append(signal)

# total counts
import matplotlib.pyplot as plt
colors = ['#724CF9','#564592','#EDF67D']
# make x axis
for signal,color in zip(signals,colors):
    x = np.linspace(-1*det_size_cm/2,det_size_cm/2,32) 
    xx = np.arctan(x/distance)

    plt.plot(np.rad2deg(xx),(signal/max(signal))/(np.sin(np.deg2rad(90)+xx)**2), color=color)
    #plt.plot(np.rad2deg(xx), (signal/max(signal)),color=color)
#plt.plot(np.rad2deg(xx), np.sin(np.deg2rad(90)+xx)**2,"#F896D8")
plt.xlabel('incident angle')
plt.ylabel('normalized intensity')

plt.savefig('../simulation-results/validating-iso/pinhole_collimation_normallized.png',dpi=300)