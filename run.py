from simulation_engine import SimulationEngine
from hits import Hits
from scattering import Scattering
from deconvolution import Deconvolution

from macros import find_disp_pos
import numpy as np

# construct = CA and TD
# source = DS and PS

simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=False)

# timepix design
det_size_cm = 1.408
pixel = 0.055  # mm

# set number of elements
rank = 17
multiplier = 14

element_size = pixel * multiplier
n_elements = (2 * rank) - 1

mask_size = element_size * n_elements

# set edge trim
trim = 9

# thickness of mask
thickness = 300  # um

distance = 2  # cm

simulation_engine.set_config(
    det1_thickness_um=300,
    det_gap_mm=30,
    win_thickness_um=100,
    det_size_cm=det_size_cm,
    n_elements=n_elements,
    mask_thickness_um=thickness,
    mask_gap_cm=distance,
    element_size_mm=element_size,
    mosaic=True,
    mask_size=mask_size
)

energy_type = "Mono"
energy_level = 100  # keV
n_particles = 1e9


simulation_engine.set_macro(
    n_particles=n_particles,
    energy_keV=[energy_type, energy_level, None],
    radius_cm=8,
    sphere=True,
)


fname = "../data/timepix_sim/iso_%d_%s_%d.csv" % (
    n_particles,
    energy_type,
    energy_level,
)

# simulation_engine.run_simulation(fname, build=True)

myhits = Hits(simulation_engine, fname)
myhits.get_det_hits()
deconvolver = Deconvolution(myhits, simulation_engine)
deconvolver.deconvolve(
    plot_deconvolved_heatmap=True,
    plot_raw_heatmap=True,
    save_raw_heatmap="../results/parameter_sweeps/timepix_sim/iso_%d_%s_%d_raw.png"
    % (n_particles, energy_type, energy_level),
    multiplier=multiplier,
    trim=trim,
    save_deconvolve_heatmap="../results/parameter_sweeps/timepix_sim/iso_%d_%s_%d_dc.png"
    % (n_particles, energy_type, energy_level),
)

deconvolver.plot_flux_signal(
    simulation_engine,
    fname="../results/parameter_sweeps/timepix_sim/iso_%d_%s_%d_flux.png"
    % (n_particles, energy_type, energy_level),
)
