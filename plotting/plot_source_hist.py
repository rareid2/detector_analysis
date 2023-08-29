import numpy as np
import csv
import sys

from plot_geant_histograms import *

histo_dir = "/home/rileyannereid/workspace/geant4/simulation-data/histo"
fname_tag = f"11-None-nosurf-pinhole"
# removing the new line characters
histos = []
for i, k in zip(
    [1, 2, 2, 2, 2, 2],
    [1, 1, 2, 3, 4, 5],
):
    with open(f"{histo_dir}/{fname_tag}_h%d_h%d.%d.csv" % (i, i, k)) as f:
        lines = [line for line in f]
    # convert histogram to data
    histo = convert_from_csv(lines, fname_tag)
    histos.append(histo)

# plot the data to
plot(histos, f"{histo_dir}/{fname_tag}.png")
