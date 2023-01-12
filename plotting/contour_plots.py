import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
# script to create a 3x2 grid of contour plots
# 3 thicknesses (cols) and 2 ranks (rows)

# loop through rank options (primes)
n_elements_original = [11, 31]  # n elements no mosaic
multipliers = [22, 8]
pixel = 0.055  # mm

element_size_mm_list = [
    pixel * multiplier for multiplier in multipliers
]  # element size in mm
n_elements_list = [
    (ne * 2) - 1 for ne in n_elements_original
]  # total number of elements

mask_size = [ne * ee for ne, ee in zip(n_elements_list, element_size_mm_list)]

# thickness of mask
thicknesses = np.logspace(2, 3.5, 5)  # im um, mask thickness
energies = []
# set up plot
fig, ax = plt.subplots(2, 2, sharex="col", sharey="row")
res_levels = np.linspace(0.0, 20, 22)
fov_levels = np.linspace(0.0, 150, 22)

for ri, rank in enumerate(n_elements_original):
    res_all = np.empty([5,12])
    fov_all = np.empty([5,12])

    for ti, thickness in enumerate(thicknesses):
        data_path = "../results/parameter_sweeps/timepix_sim/res/%d_%d.txt" % (
            thickness,
            rank,
        )
        data = np.loadtxt(data_path)
        distance = [d[0] for d in data]
        res = [d[1] for d in data]

        data_path = "../results/parameter_sweeps/timepix_sim/fov/%d_%d.txt" % (
            thickness,
            rank,
        )
        data = np.loadtxt(data_path)
        fov = [2*d[1] for d in data]

        res_all[ti,:] = res
        fov_all[ti,:] = fov

    FS = ax[1,ri].contourf(np.array(distance), thicknesses, fov_all, levels=fov_levels, cmap=cm.coolwarm_r, extend='max')
    CS = ax[0,ri].contourf(np.array(distance), thicknesses, res_all, levels=res_levels, cmap=cm.coolwarm, extend='max')
    ax[0,ri].set_yscale('log')
    ax[1,ri].set_yscale('log')


colorbar = plt.colorbar(CS)
colorbar = plt.colorbar(FS)

plt.savefig(
    "/home/rileyannereid/workspace/geant4/detector_analysis/plotting/contour_plot.png",
    dpi=1000,
)
