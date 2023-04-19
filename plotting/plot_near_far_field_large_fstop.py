import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib
import numpy as np
from plot_settings import *

    """plot near vs far field simulation results for a large range of f-stops
    """

# loop through rank options (primes)
n_elements_original = [11,31]  # n elements no mosaic
multipliers = [22, 8]
pixel = 0.055  # mm
det_size_cm = 1.408

element_size_mm_list = [
    pixel * multiplier for multiplier in multipliers
]  # element size in mm
n_elements_list = [
    (ne * 2) - 1 for ne in n_elements_original
]  # total number of elements

mask_size = [ne * ee for ne, ee in zip(n_elements_list, element_size_mm_list)]

# thickness of mask
thicknesses = np.logspace(2, 3.5, 5)  # im um, mask thickness
thickness = thicknesses[2]
energies = [0.35, 0.65, 1.2, 3.1, 10]

# set up plot
fig, ax = plt.subplots(1, 1)

params = {'axes.labelsize': 16,
          'axes.titlesize': 16}
plt.rcParams.update(params)

linestyles = ['dotted','-']

distances = np.flip(np.linspace(0.1 * 2 * det_size_cm, 10 * 2 * det_size_cm, 50))
# just grab the last 2
distances = [distances[0], distances[-1]]

hex_list = ["#392F5A","#9DD9D2","#FFB703","#FB8500","#F15025"]

ci = 0
for ri, rank in enumerate(n_elements_original):
    for start_distance in distances:
        distance = start_distance - ((150 + (thickness/2))*1e-4)

        data_path ="../simulation_results/parameter_sweeps/prospectus_plots/fwhm/%d_%d_%.2f.txt" % (thickness, rank, distance)
        data = np.loadtxt(data_path)
        fstop = np.array(distance)/(2*det_size_cm)

        source_distance = [d[0] for d in data]
        fwhm = [d[1] for d in data]

        for si, sd in enumerate(source_distance):
            if rank == 31 and start_distance > 10 and sd < 150:
                fwhm[si] = np.nan
                
        ax.plot(
            np.array(source_distance)/1e3, fwhm, color=hex_list[ci], linestyle = linestyles[ri],linewidth=1.5,label='rank %d F-number %.1f' % (rank,fstop))
        ci+=1


plt.legend()
ax.grid(which="major", color="#DDDDDD", linewidth=0.8)
ax.grid(which="minor", color="#EEEEEE", linestyle=":", linewidth=0.5)
ax.minorticks_on()
ax.set_ylabel('FWHM [pixels]')
ax.set_xlabel('source distance [km]')
plt.savefig(
"plotting/nearfar_prospectus.png",bbox_inches='tight',
dpi=1000
)
