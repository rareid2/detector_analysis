import sys

sys.path.insert(1, "../detector_analysis")
from hits import Hits
from deconvolution import Deconvolution
from simulation_engine import SimulationEngine
import numpy as np

    """create example plots for resolution testing for prospectus
    """

# timepix design
det_size_cm = 1.408
# loop through rank options (primes)
n_elements = 31
multiplier=8
pixel = 0.055  # mm
trim = 4
thickness = 1333
distance = 1.34
res_deg = 4.7
simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=True)
simulation_engine.set_config(
    det1_thickness_um=300,
    det_gap_mm=30,
    win_thickness_um=100,
    det_size_cm=det_size_cm,
    n_elements=61,
    mask_thickness_um=thickness,
    mask_gap_cm=distance,
    element_size_mm= pixel * multiplier,
    mosaic=True,
    mask_size=round(pixel*multiplier * 31, 2),
)
fov_edge = 22.67
fname = (
    "../simulation_data/timepix_sim/prospectus_plots/fov/%d_%d_%.2f_%.2f_%.2f.csv"
    % (thickness, n_elements, distance, res_deg, fov_edge)
)

myhits = Hits(fname=fname, experiment=False)
myhits.get_det_hits()
deconvolver = Deconvolution(myhits, simulation_engine)
# res is changed from true or false to a peak location
res, _, _, _ = deconvolver.deconvolve(
    plot_deconvolved_heatmap=True,
    save_raw_heatmap="../simulation_results/parameter_sweeps/prospectus_plots/fov/new_%d_%d_%.2f_%.2f_%.2f_raw.png"
    % (thickness, n_elements, distance, res_deg, fov_edge),
    plot_raw_heatmap=True,
    downsample=1,
    trim=trim,
    save_deconvolve_heatmap="../simulation_results/parameter_sweeps/prospectus_plots/fov/new_%d_%d_%.2f_%.2f_%.2f.png"
    % (thickness, n_elements, distance, res_deg, fov_edge),
    save_peak="../simulation_results/parameter_sweeps/prospectus_plots/fov/new_%d_%d_%.2f_%.4f_%.2f_peak.png"
    % (thickness, n_elements, distance, res_deg, fov_edge),
    plot_signal_peak=True,
    plot_conditions=True,
    check_resolved=True,
    condition="half_val",
)
