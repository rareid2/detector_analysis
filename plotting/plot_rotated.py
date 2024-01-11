import numpy as np
import matplotlib.pyplot as plt
from plot_settings import *


fname1 = "/home/rileyannereid/workspace/geant4/simulation-results/rings/final_image/73-2-33p02-deg-circle-rotate-0_1.16E+09_Mono_100_dc.txt"
fname2 = "/home/rileyannereid/workspace/geant4/simulation-results/rings/final_image/73-2-33p02-deg-circle_1.16E+09_Mono_100_dc.txt"

r1 = np.loadtxt(fname1)
r2 = np.loadtxt(fname2)

plt.imshow(r2+r1,cmap=cmap)
plt.savefig("/home/rileyannereid/workspace/geant4/simulation-results/rings/final_image/roated_combined.png",dpi=300)
