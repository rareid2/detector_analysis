import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.io import savemat

import cmocean

cmap = cmocean.cm.thermal

txt_folder = "/home/rileyannereid/workspace/geant4/simulation-results/59-fwhm-3/"
hz = np.loadtxt(f"{txt_folder}xy-hits.txt")
hc = np.loadtxt(f"{txt_folder}0-hits.txt")
hc = float(hc)

zz = np.loadtxt(f"{txt_folder}xy-signal.txt")
zc = np.loadtxt(f"{txt_folder}0-signal.txt")

# zz = np.loadtxt(f"{txt_folder}xy-fwhm.txt")
# zc = np.loadtxt(f"{txt_folder}0-fwhm.txt")

zc = float(zc)

zz = np.array([(z / zc) * 1 / (h / hc) for z, h in zip(zz, hz)])
# zz = np.array([((zc - z) / zc) for z in zz])
zz = np.insert(zz, 0, 0)

reverse_zz = np.flip(zz)
stacked_zz = np.hstack((reverse_zz, zz[1:]))

main_diagonal = [[i, i] for i in range(0, (59 // 2), 2)]
main_diagonal = main_diagonal[:-1]
main_diagonal = np.vstack((-1 * np.flip(main_diagonal[1:]), main_diagonal))
diagonal_radial = np.array([np.sqrt(md[0] ** 2 + md[1] ** 2) for md in main_diagonal])


# okay now we have the function
def polynomial_function(x, *coefficients):
    return np.polyval(coefficients, x)


degree = 2
initial_guess = np.ones(degree + 1)  # Initial guess for the polynomial coefficients
params, covariance = curve_fit(
    polynomial_function, diagonal_radial, stacked_zz, p0=initial_guess
)
width, height = 59, 59
x = np.linspace(0, width - 1, width)
y = np.linspace(0, height - 1, height)
x, y = np.meshgrid(x, y)
center_x, center_y = width // 2, height // 2
distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

new_z = polynomial_function(distance, *params)

plt.imshow(new_z, cmap=cmap)
plt.colorbar(label="pixels")
output_name = "signal"
plt.savefig(f"{txt_folder}{output_name}_2D.png", dpi=300)
