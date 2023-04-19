import matplotlib.pyplot as plt
import numpy as np
from plot_settings import *

"""
plotting script to create polar plots of resolution across the field of view used for propsectus
"""

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
thicknesses = np.logspace(2, 3.5, 5)  # im um, mask thickness
energies = [0.35, 0.65, 1.2, 3.1, 10]
# set up plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

params = {'axes.labelsize': 36,
          'axes.titlesize': 16}
plt.rcParams.update(params)
ax.set_rlim(bottom=0.28, top=0.255)

thick_color = ["#03045e","#0077b6","#00b4d8","#90e0ef","#caf0f8"]
thick_color.reverse()
thick_color = hex_colors[::int(np.ceil( len(hex_colors) / len(energies) ))]

for ti, thickness in enumerate(thicknesses):
    for ri, rank in enumerate(n_elements_original):

        data_path = "../simulation-results/prospectus/fov/%d_%d_res_2.txt" % (
            thickness,
            rank,
        )
        data = np.loadtxt(data_path)

        fov1 = np.linspace(0,25,10)
        fov2 = -1*fov1[1:]
        fov2 = np.flip(fov2)
        fov = []
        fov.extend(fov2)
        fov.extend(fov1)
        fov = np.deg2rad(fov)


        #if thickness > 3000:
        #    res = [d[1] for d in data[:8]]
        #    res.append(np.nan)
        #    res.append(np.nan)

        #    res.reverse()
        #    newres = [d[1] for d in data[:8]]
        #    newres.append(np.nan)
        #    newres.append(np.nan)

        #    newres = newres[1:]
        #    res.extend(newres)
        #else:
        res = [d[1] for d in data]
        res.reverse()
        newres = [d[1] for d in data]
        newres = newres[1:]
        res.extend(newres)
 

    plt.plot(fov, res, color=thick_color[ti], marker="o", linewidth=3)


plt.xlim([np.deg2rad(-27),np.deg2rad(27)])
plt.xticks(np.deg2rad([-25,-12.5,0,12.5,25]))

plt.savefig(
    "../simulation-results/prospectus/fov/polar_plot_prospectus_5deg.png",
    dpi=1000
)
