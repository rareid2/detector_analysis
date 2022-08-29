from simulation_engine import SimulationEngine
from hits import Hits
from scattering import Scattering
from deconvolution import Deconvolution

# example run for coded aperture with point sources

# branch names:
# construct = CA and TD
# source = DS and PS

simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=True)
simulation_engine.set_config(
    det1_thickness_um=140,
    det_gap_mm=30,
    win_thickness_um=100,
    det_size_cm=4.422,
    n_elements=133,
    mask_thickness_um=400,
    mask_gap_cm=3,
    element_size_mm=0.66,
)

simulation_engine.set_macro(n_particles=100000, energy_keV=[500])
# fname = "../data/hits_1e9_3_400_sin.csv"
fname = "../data/test.csv"
#simulation_engine.run_simulation(fname, build=False)

myhits = Hits(simulation_engine, fname)
myhits.get_det_hits()
deconvolver = Deconvolution(myhits, simulation_engine)
deconvolver.deconvolve(
    plot_deconvolved_heatmap=True,
    plot_raw_heatmap=False,
    multiplier=12,
    save_deconvolve_heatmap="test_dc.png",
    save_peak="test.png",
    plot_signal_peak=True,
)
