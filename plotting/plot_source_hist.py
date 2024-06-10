import numpy as np
import csv
import sys
# plot the data to
import matplotlib.pyplot as plt
from plot_geant_histograms import *

histo_dir = "/home/rileyannereid/workspace/geant4"
fname_tag = f"src_spectrum"
# removing the new line characters
histos = []
for i, k in zip(
    [1],
    [1],
):
    with open(f"{histo_dir}/{fname_tag}_h%d_h%d.%d.csv" % (i, i, k)) as f:
        lines = [line for line in f]
    # convert histogram to data
    histo = convert_from_csv(lines, fname_tag)
    histos.append(histo)

fname = f"{histo_dir}/{fname_tag}.png"

figure = plt.figure()
figure.tight_layout(pad=0.5)
histo = histos[0]
bins = np.array(histo["bins"][0])

x = (histo["bin_edges"][0] + histo["bin_edges"][1]) / 2
xerr = (histo["bin_edges"][0] - histo["bin_edges"][1]) / 2
print(histo["bin_edges"][0])
# Bins: Entries,Sum(W),Sum(W**2),...
y = bins[:, 1]
# Error on bin content: Sqrt(Sum W**2 - (Sum W)**2/N)
_entries = np.sum(bins[:, 0])
yerr = np.sqrt(bins[:, 2] - bins[:, 1] ** 2 / _entries)
plt.xlim(x[0] - xerr[0], x[-1] + xerr[-1])

plt.plot(x, y, markersize=3)
print(np.sum(y))
print("finished plotting")
plt.savefig(fname)
