from simulation_engine import SimulationEngine
from hits import Hits
from scattering import Scattering
from deconvolution import Deconvolution

# example run for coded aperture with point sources

# branch names:
# construct = CA and TD
# source = DS and PS

simulation_engine = SimulationEngine(construct="CA", source="DS", write_files=True)
simulation_engine.set_config(
    det1_thickness_um=140,
    det_gap_mm=30,
    win_thickness_um=100,
    det_size_cm=4.422,
    n_elements=1,
    mask_thickness_um=400,
    mask_gap_cm=1,
    element_size_mm=1,
    mosaic=False,
    mask_size=87,
)

sims = [0, 1, 2, 3]
sim_str = ["90","sine", "sine_sq", "tri"]

for s, ss in zip(sims, sim_str):
    simulation_engine.set_macro(n_particles=100000, energy_keV=500, PAD_run=s)
    fname = "../data/pinhole/pinhole_%s.csv" % (ss)
    #simulation_engine.run_simulation(fname, build=True)

    myhits = Hits(simulation_engine, fname)
    myhits.get_det_hits()

    deconvolver = Deconvolution(myhits, simulation_engine)
    deconvolver.multiplier = 1

    # shift origin
    deconvolver.shift_pos()

    # get heatmap
    deconvolver.get_raw()

    deconvolver.plot_heatmap(deconvolver.raw_heatmap, save_name="../results/pinhole/pinhole_%s.png" % (ss))