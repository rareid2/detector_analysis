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
rank = 31
pixels_downsample = 8

element_size = pixel * pixels_downsample
n_elements = (2 * rank) - 1

mask_size = element_size * n_elements
# set edge trim - can't use all pixels to downsample to integer amount
trim = 4
mosaic = True
# -------------------------------------

# -------- pinhole set up -------------
"""
rank = 1
element_size = 1.76 # mm
pixels_downsample = int(element_size / pixel)
n_elements = rank
mask_size = det_size_cm
trim = None
mosaic = False
"""
# -------------------------------------

# thickness of mask
thickness = 500  # um

# focal length
distance = 2.478  # cm

# number of particles to simulate
n_particles = 1000000

# --------------set up simulation---------------
simulation_engine.set_config(
    det1_thickness_um=300,
    det_gap_mm=30, # gap between first and second (unused detector)
    win_thickness_um=100, # window is not actually in there
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

#simulation_engine.set_macro(
#    n_particles=n_particles,
#    energy_keV=[energy_type, energy_level, None],
#    radius_cm=8,
#    sphere=True,
#    progress_mod=int(n_particles/10) # set with 10 steps
#)

simulation_engine.set_macro(
    n_particles=n_particles,
    energy_keV=[energy_level],
    directions=[0],
    positions=[[0,0,-500]],
    sphere=False,
    progress_mod=int(n_particles/10) # set with 10 steps
)

# --------------set up data naming---------------
fname_tag = 'bleh'
fname = "../data/timepix_sim/%s_%d_%s_%d.csv" % (
    fname_tag,
    n_particles,
    energy_type,
    energy_level,
)

# --------------RUN---------------
#simulation_engine.run_simulation(fname, build=True)

# ---------- process results -----------
myhits = Hits(simulation_engine, fname)
myhits.get_det_hits()

# deconvolution steps
deconvolver = Deconvolution(myhits, simulation_engine)

deconvolver.deconvolve(
    plot_deconvolved_heatmap=True,
    plot_raw_heatmap=True,
    save_raw_heatmap="../results/parameter_sweeps/timepix_sim/%s_%d_%s_%d_raw.png"
    % (fname_tag, n_particles, energy_type, energy_level),
    save_deconvolve_heatmap="../results/parameter_sweeps/timepix_sim/%s_%d_%s_%d_dc.png"
   % (fname_tag, n_particles, energy_type, energy_level),
    downsample=pixels_downsample,
    trim=trim,
)
