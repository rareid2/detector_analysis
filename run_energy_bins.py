from simulation_engine import SimulationEngine
from hits import Hits
from scattering import Scattering
from deconvolution import Deconvolution
from scipy import interpolate
from macros import find_disp_pos
import numpy as np
import os
import subprocess
import copy

from scipy.io import savemat
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from plotting.plot_settings import *
from plotting.plot_geant_histograms import *

# construct = CA and TD
# source = DS and PS

simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=True)

txt = False
simulate = False
nthreads = 14

trim = None
mosaic = True
multiplier = 1

# designs to evaluate
det_size_cm = 4.956  # cm
pixel = 0.84  # mm
n_elements_original = 59
thickness = 100
distance = 0.923
radius = 7.3

pixel_size = pixel * 0.1
element_size = pixel * multiplier
n_elements = (2 * n_elements_original) - 1
mask_size = element_size * n_elements

n_particles = 1e8

# --------------set up simulation---------------
simulation_engine.set_config(
    det1_thickness_um=500,
    det_gap_mm=30,  # gap between first and second (unused detector)
    win_thickness_um=100,  # window is not actually in there
    det_size_cm=det_size_cm,
    n_elements=n_elements,
    mask_thickness_um=thickness,
    mask_gap_cm=distance,
    element_size_mm=element_size,
    mosaic=mosaic,
    mask_size=mask_size,
    radius_cm=radius,
)

# --------------set up source---------------
energy_type = "Pow"
energy_min = 0.1
energy_max = 3

# --------------set up data naming---------------
fname_tag = f"{n_elements_original}-{distance}"

fname = f"../simulation-data/energy-spectrum/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_max}.csv"

if txt:
    fname = f"../simulation-results/energy-spectrum/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_max}_raw.txt"

simulation_engine.set_macro(
    n_particles=int(n_particles),
    energy_keV=[energy_type, energy_min, energy_max],
    surface=True,
    progress_mod=int(n_particles / 10),  # set with 10 steps
    fname_tag=fname_tag,
    confine=False,
    detector_dim=det_size_cm,
    theta=None,
    ring=False,
    radius_cm=radius,
)

# --------------RUN---------------
if simulate:
    simulation_engine.run_simulation(fname, build=False, rename=True)

# ---------- process results -----------
if not txt:
    for hi in range(nthreads):
        print(hi)
        fname_hits = fname[:-4] + "-{}.csv".format(hi)
        myhits = Hits(fname=fname_hits, experiment=False, txt_file=txt)
        hits_dict, sec_brehm, sec_e = myhits.get_det_hits(
            remove_secondaries=False, second_axis="y", get_edep=True, det_thick_cm=0.05
        )

        if hi != 0:
            # update fields in hits dict
            myhits.hits_dict["Position"].extend(hits_copy.hits_dict["Position"])
            myhits.hits_dict["Energy"].extend(hits_copy.hits_dict["Energy"])
            myhits.hits_dict["E0"].extend(hits_copy.hits_dict["E0"])
            myhits.hits_dict["Edep"].extend(hits_copy.hits_dict["Edep"])

            hits_copy = copy.copy(myhits)
        else:
            hits_copy = copy.copy(myhits)

else:
    myhits = Hits(fname=fname, experiment=False, txt_file=txt)


# now we sort by energy
histo_dir = "/home/rileyannereid/workspace/geant4/simulation-data/energy-spectrum"
histo_results_dir = (
    "/home/rileyannereid/workspace/geant4/simulation-results/energy-spectrum"
)

fname_tag = f"src_spectrum"
i = 1
k = 1
with open(f"{histo_dir}/{fname_tag}_h%d_h%d.%d.csv" % (i, i, k)) as f:
    lines = [line for line in f]
# convert histogram to data
histo = convert_from_csv(lines, fname_tag)

figure = plt.figure()

bins = np.array(histo["bins"][0])
x = (histo["bin_edges"][0] + histo["bin_edges"][1]) / 2
xerr = (histo["bin_edges"][0] - histo["bin_edges"][1]) / 2

y = bins[:, 1]
_entries = np.sum(bins[:, 0])
yerr = np.sqrt(bins[:, 2] - bins[:, 1] ** 2 / _entries)
plt.xlim(x[0] - xerr[0], x[-1] + xerr[-1])
plt.plot(x, y, markersize=3)

fname = f"{histo_results_dir}/source_spectrum.png"
plt.savefig(fname)
plt.clf()

simulated_bins = histo["bin_edges"][0]
simulated_bins = np.append(simulated_bins, energy_max)

# for each incident particle - where did it deposut energy?
n_inst_bins = 15
instrument_bins = np.logspace(np.log10(energy_min), np.log10(energy_max), n_inst_bins)
import cmocean

cmap = cmocean.cm.thermal

counts = np.zeros((len(instrument_bins), len(x)))

for hit_n, hit in enumerate(myhits.hits_dict["E0"]):
    # find which y this is at
    for si, sb in enumerate(simulated_bins[:-1]):
        if sb <= hit < simulated_bins[si + 1]:
            n_sim = si

    # now we know which n_sim, what energy did it deposit?
    hit_energy = myhits.hits_dict["Edep"][hit_n]
    # now we know which simulated bin it is in
    # which instrument bin does that fall into???
    for bn, ib in enumerate(instrument_bins[:-1]):
        if ib <= hit_energy < instrument_bins[bn + 1]:
            # now we have the right instrument bin to include it in
            counts[bn, n_sim] += 1

gf_factor = 4 * np.pi**2 * radius**2
for yi, yy in enumerate(counts[:-1]):
    linex = []
    liney = []
    for xi, xx in enumerate(x):
        linex.append(simulated_bins[xi])
        linex.append(simulated_bins[xi + 1])
        liney.append((yy[xi] / y[xi]) * gf_factor)
        liney.append((yy[xi] / y[xi]) * gf_factor)
    plt.plot(
        linex,
        liney,
        color=cmap(yi / n_inst_bins),
        label=f"{round(instrument_bins[yi],2)}-{round(instrument_bins[yi+1],2)} keV",
    )
plt.legend(fontsize="8", ncol=2)
plt.xlabel("Incident Energy [keV]")
plt.ylabel("Geometric Factor [cm^2 sr]")
plt.savefig(f"{histo_results_dir}/gf_dep.png", dpi=500)
