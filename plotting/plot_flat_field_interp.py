import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

# Load x and y values from separate text files
txt_folder = "./results/61-2-400/"

hits = False
remove_edges = True

if remove_edges:
    edges_str = "edges-removed"
else:
    edges_str = "edges-inc"
if hits:
    hits_str = "instrument-only"
else:
    hits_str = "shielding-inc"

data_product = "fwhm"
output_name = f"{txt_folder}{data_product}_interp_grid_{hits_str}_{edges_str}"

# load data
zx = np.loadtxt(f"{txt_folder}x-{data_product}.txt")
zy = np.loadtxt(f"{txt_folder}y-{data_product}.txt")
zz = np.loadtxt(f"{txt_folder}xy-{data_product}.txt")

zc = np.loadtxt(f"{txt_folder}0-{data_product}.txt")
zc = float(zc)

# hits - if we want to see flat field of instrument response (only)
if hits and data_product == "signal":
    hx = np.loadtxt(f"{txt_folder}x-hits.txt")
    hy = np.loadtxt(f"{txt_folder}y-hits.txt")
    hz = np.loadtxt(f"{txt_folder}xy-hits.txt")

    hc = np.loadtxt(f"{txt_folder}0-hits.txt")
    hc = float(hc)

    zx = np.array([(z/zc) * 1 / (h/hc) for z, h in zip(zx, hx)])
    zy = np.array([(z/zc) * 1 / (h/hc) for z, h in zip(zy, hy)])
    zz = np.array([(z/zc) * 1 / (h/hc) for z, h in zip(zz, hz)])
    zx = np.insert(zx, 0, 1)
    zy = np.insert(zy, 0, 1)
    zz = np.insert(zz, 0, 1) 


elif data_product == "signal":
    zx = np.array([(z/zc) for z in zx])
    zy = np.array([(z/zc) for z in zy])
    zz = np.array([(z/zc) for z in zz]) 
    zx = np.insert(zx, 0, 1)
    zy = np.insert(zy, 0, 1)
    zz = np.insert(zz, 0, 1) 


else:
    # add in central fwhm
    zx = np.insert(zx, 0, zc)
    zy = np.insert(zy, 0, zc)
    zz = np.insert(zz, 0, zc)

if remove_edges:
    zx = zx[:-1]
    zy = zy[:-1]
    zz = zz[:-1]

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

xcrs = np.array([(xx, 0) for xx in x])
ycrs = np.array([(0, yy) for yy in y])

# create diagonal coordinates
main_diagonal = np.array([(i - 30, i - 30) for i in range(0, 61, 2)])
other_diagonal = np.array([(i - 30, 30 - i) for i in range(0, 61, 2)])

# Define the grid where you want to interpolate
x_new = np.linspace(-30, 30, 122)
y_new = np.linspace(-30, 30, 122)

x_new_grid, y_new_grid = np.meshgrid(x_new, y_new)

diagonals = np.concatenate((main_diagonal, other_diagonal))
points = np.concatenate((xcrs, ycrs, diagonals))
values = np.concatenate((stacked_zx, stacked_zy, stacked_zz, stacked_zz))

# Perform interpolation
z_new = interpolate.griddata(points, values, (x_new_grid, y_new_grid), method="cubic")
np.savetxt(
    f"{output_name}.txt",
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
plt.savefig(f"{output_name}.png", dpi=300)

# Create a 3D figure
plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Create the surface plot!
surface = ax.plot_surface(x_new_grid, y_new_grid, z_new, cmap="viridis")

# Add a color bar for reference
fig.colorbar(surface)

# Show the plot
ax.set_xlabel("pixel")
ax.set_ylabel("pixel")
ax.set_zlabel("signal")
plt.savefig(f"{output_name}_3D.png", dpi=300)