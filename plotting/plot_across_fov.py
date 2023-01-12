import matplotlib.pyplot as plt
import numpy as np


# loop through rank options (primes)
n_elements_original = [31]  # n elements no mosaic
multipliers = [8]
pixel = 0.055  # mm

element_size_mm_list = [
    pixel * multiplier for multiplier in multipliers
]  # element size in mm
n_elements_list = [
    (ne * 2) - 1 for ne in n_elements_original
]  # total number of elements

mask_size = [ne * ee for ne, ee in zip(n_elements_list, element_size_mm_list)]

# thickness of mask
thicknesses = [100,237,562,1333,3150]  # im um, mask thickness
energies = [10]
# set up plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

params = {'axes.labelsize': 30,
          'axes.titlesize': 16}
plt.rcParams.update(params)

thick_color = ["#03045e","#0077b6","#00b4d8","#90e0ef","#caf0f8"]
thick_color.reverse()


for ti, thickness in enumerate(thicknesses):
    for ri, rank in enumerate(n_elements_original):

        data_path = "../results/parameter_sweeps/timepix_sim/signal_fov/%d_%d.txt" % (
            thickness,
            rank,
        )

        data = np.loadtxt(data_path)
        fov = np.array([25,22.5,20,17.5,15,12.5,10,7.5,5,2.5,0,-2.5,-5,-7.5,-10,-12.5,-15,-17.5,-20,-22.5,-25])
        fov = np.deg2rad(fov)
        res = [d[1] for d in data]
        res = [r/res[0] for r in res]
        res.reverse()

        newres = [d[1] for d in data]
        newres = [r/newres[0] for r in newres]
        newres = newres[1:]
        res.extend(newres)
        plt.plot(fov, res, color=thick_color[ti], marker="o")


plt.xlim([np.deg2rad(-27),np.deg2rad(27)])
plt.xticks(np.deg2rad([-25,-12.5,0,12.5,25]))

plt.savefig(
    "/home/rileyannereid/workspace/geant4/detector_analysis/plotting/polar_plot.png",
    dpi=1000
)
