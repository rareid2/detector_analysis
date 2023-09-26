import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

# Load x and y values from separate text files
txt_folder = "/home/rileyannereid/workspace/geant4/simulation-results/aperture-collimation/geom-correction/"
zx = np.loadtxt(f"{txt_folder}x-sig.txt")
zy = np.loadtxt(f"{txt_folder}y-sig.txt")
zz = np.loadtxt(f"{txt_folder}xy-sig.txt")

# hits - if we want to see flat field of instrument response (only)
"""
hx = np.loadtxt(f"{txt_folder}x-hits-norm.txt")
hy = np.loadtxt(f"{txt_folder}y-hits-norm.txt")
hz = np.loadtxt(f"{txt_folder}xy-hits-norm.txt")

zx = np.array([z * 1 / h for z, h in zip(zx, hx)])
zy = np.array([z * 1 / h for z, h in zip(zy, hy)])
zz = np.array([z * 1 / h for z, h in zip(zz, hz)])
"""
zx = zx[:-1]
zy = zy[:-1]
zz = zz[:-1]

zx = np.insert(zx, 0, 1)
zy = np.insert(zy, 0, 1)
zz = np.insert(zz, 0, 1)

# replace the edge value
x_values = np.linspace(0, 28, 15)

# Perform cubic interpolation using CubicSpline
cubic_spline = interpolate.CubicSpline(x_values, zx)
extrap_x = cubic_spline(30)

cubic_spline = interpolate.CubicSpline(x_values, zy)
extrap_y = cubic_spline(30)

cubic_spline = interpolate.CubicSpline(x_values, zz)
extrap_z = cubic_spline(30)

# insert them
zx = np.append(zx, extrap_x)
zy = np.append(zy, extrap_y)
zz = np.append(zz, extrap_z)

# Create an array in reverse order
reverse_zx = np.flip(zx)
reverse_zy = np.flip(zy)
reverse_zz = np.flip(zz)

# Stack the input and reverse arrays together
stacked_zx = np.hstack((reverse_zx, zx[1:]))
stacked_zy = np.hstack((reverse_zy, zy[1:]))
stacked_zz = np.hstack((reverse_zz, zz[1:]))

# create coordinates for x and y
x = np.linspace(-30, 30, 31)
y = np.linspace(-30, 30, 31)

# x = x[1:-1]
# y = y[1:-1]

xcrs = np.array([(xx, 0) for xx in x])
ycrs = np.array([(0, yy) for yy in y])

# create diagonal coordinates
main_diagonal = np.array([(i - 30, i - 30) for i in range(0, 61, 2)])
other_diagonal = np.array([(i - 30, 30 - i) for i in range(0, 61, 2)])

# exclude corner points for now
# main_diagonal = main_diagonal[1:-1]
# other_diagonal = other_diagonal[1:-1]

# Define the grid where you want to interpolate
x_new = np.linspace(-30, 30, 61)
y_new = np.linspace(-30, 30, 61)

# x_new = x_new[2:-2]
# y_new = y_new[2:-2]

x_new_grid, y_new_grid = np.meshgrid(x_new, y_new)

diagonals = np.concatenate((main_diagonal, other_diagonal))
points = np.concatenate((xcrs, ycrs, diagonals))
values = np.concatenate((stacked_zx, stacked_zy, stacked_zz, stacked_zz))

# Perform interpolation
z_new = interpolate.griddata(points, values, (x_new_grid, y_new_grid), method="cubic")
np.savetxt(
    f"{txt_folder}interp_grid.txt",
    z_new,
)

# Create a plot to visualize the original data and the interpolated surface
plt.figure(figsize=(9, 4))
plt.subplot(1, 2, 1)
plt.title("Original Data")
plt.scatter(xcrs[:, 0], xcrs[:, 1], c=stacked_zx, cmap="viridis", marker="o")
plt.scatter(ycrs[:, 0], ycrs[:, 1], c=stacked_zy, cmap="viridis", marker="o")
plt.scatter(
    main_diagonal[:, 0], main_diagonal[:, 1], c=stacked_zz, cmap="viridis", marker="o"
)
plt.scatter(
    other_diagonal[:, 0], other_diagonal[:, 1], c=stacked_zz, cmap="viridis", marker="o"
)

plt.colorbar(label="Z Value")
plt.xlabel("X")
plt.ylabel("Y")

plt.subplot(1, 2, 2)
plt.title("Interpolated Surface")
plt.contourf(x_new_grid, y_new_grid, z_new, cmap="viridis", levels=100)
plt.colorbar(label="Z Value")
plt.xlabel("X")
plt.ylabel("Y")

plt.tight_layout()
# plt.gca().set_aspect("equal")
plt.savefig(f"{txt_folder}interpolation-corrected.png", dpi=300)
