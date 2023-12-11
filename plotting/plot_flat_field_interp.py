import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.io import savemat

# Load x and y values from separate text files
txt_folder = "/home/rileyannereid/workspace/geant4/simulation-results/67-2-fwhm-delta/"

maxpixel = 96
num_points = 16
maxgrid  = 192
px_int = 6

hits = True
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

    zx = np.array([(z / zc) * 1 / (h / hc) for z, h in zip(zx, hx)])
    zy = np.array([(z / zc) * 1 / (h / hc) for z, h in zip(zy, hy)])
    zz = np.array([(z / zc) * 1 / (h / hc) for z, h in zip(zz, hz)])
    zx = np.insert(zx, 0, 1)
    zy = np.insert(zy, 0, 1)
    zz = np.insert(zz, 0, 1)

elif data_product == "signal":
    zx = np.array([(z / zc) for z in zx])
    zy = np.array([(z / zc) for z in zy])
    zz = np.array([(z / zc) for z in zz])
    zx = np.insert(zx, 0, 1)
    zy = np.insert(zy, 0, 1)
    zz = np.insert(zz, 0, 1)

else:
    # add in central fwhm or central hits ratio
    zx = np.insert(zx, 0, zc)
    zy = np.insert(zy, 0, zc)
    zz = np.insert(zz, 0, zc)

if remove_edges:
    zx = zx[:-1]
    zy = zy[:-1]
    zz = zz[:-1]
    maxpixel -= px_int

# Create an array in reverse order
reverse_zx = np.flip(zx)
reverse_zy = np.flip(zy)
reverse_zz = np.flip(zz)

# Stack the input and reverse arrays together
stacked_zx = np.hstack((reverse_zx, zx[1:]))
stacked_zy = np.hstack((reverse_zy, zy[1:]))
stacked_zz = np.hstack((reverse_zz, zz[1:]))

# we only need stacked zz now
main_diagonal = np.array([(i - maxpixel, i - maxpixel) for i in range(0, maxgrid-px_int, px_int)])
diagonal_radial = np.array([np.sqrt(md[0]**2 + md[1]**2) for md in main_diagonal])

# okay now we have the function
def polynomial_function(x, *coefficients):
    return np.polyval(coefficients, x)

# Fit the curve with a polynomial of degree 3 (you can adjust the degree)
degree = 3
initial_guess = np.ones(degree + 1)  # Initial guess for the polynomial coefficients
params, covariance = curve_fit(polynomial_function, diagonal_radial, stacked_zz, p0=initial_guess)

width, height = 201, 201
x = np.linspace(0, width - 1, width)
y = np.linspace(0, height - 1, height)
x, y = np.meshgrid(x, y)

# Calculate the distance from the center of the grid
center_x, center_y = width // 2, height // 2
distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)

new_z = polynomial_function(distance, *params)

np.savetxt(
    f"{output_name}_1d.txt",
    new_z,
)

# Create a 3D figure
plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Create the surface plot!
surface = ax.plot_surface(x, y, new_z, cmap="viridis")

# Add a color bar for reference
fig.colorbar(surface)

# Show the plot
ax.set_xlabel("pixel")
ax.set_ylabel("pixel")
ax.set_zlabel("signal")
plt.savefig(f"{output_name}_3D.png", dpi=300)
plt.clf()

plt.imshow(new_z)
plt.colorbar(label="pixels")
plt.savefig(f"{output_name}_2D.png", dpi=300)

data = {"fwhm": new_z}

# Specify the file path where you want to save the .mat file
file_path = "fwhm.mat"

# Save the grid data as a .mat file
savemat(file_path, data)
