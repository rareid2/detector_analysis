import numpy as np
import matplotlib.pyplot as plt
from plot_settings import *
# general detector design
det_size_cm = 2.19  # cm
pixel = 0.1  # mm
pixel_size = pixel * 0.1

energy_type = "Mono"
energy_level = 100  # keV

# thickness of mask
thickness = 400  # um

# focal length
distance = 2  # cm

thetas = [2.865, 11.305, 19.295, 26.565, 33.025, 33.025, 33.025]

# Set the number of rows and columns in the subplot grid
rows, cols = 1, 3

# Create subplots and plot arrays
fig, axes = plt.subplots(rows, cols, figsize=(7, 2),sharex=True)

# ------------------- simulation parameters ------------------
for n in [1]:
    if n == 0:
        area = 1.095
        n_elements_original = 1
    else:
        area = 2.445
        n_elements_original = 73

    for ii, theta in enumerate(thetas[-3:]):

        n_particles = int((3e8* (area * 2) ** 2) * (1 - np.cos(np.deg2rad(theta))))

        formatted_theta = "{:.0f}p{:02d}".format(int(theta), int((theta % 1) * 100))
        if n == 0:
            fname_tag = f"{n_elements_original}-{distance}-{formatted_theta}-deg-circle-pinhole-c"
        else:
            if ii > 0:
                fname_tag = f"{n_elements_original}-{distance}-{formatted_theta}-deg-circle-rotate-0"
            else:
                fname_tag = f"{n_elements_original}-{distance}-{formatted_theta}-deg-circle"

        fname = f"../simulation-results/rings/final_image/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_dc.txt"

        if ii != 2:
            array = np.loadtxt(fname) 

            ax = axes[ii]

            cax = ax.imshow(array, cmap=cmap)
            cbar = fig.colorbar(cax,ax=ax,shrink=0.5,orientation="horizontal",pad=0.01)
            cbar.formatter.set_powerlimits((0,0))
            ax.axis('off')
        else:
            ax = axes[ii]

            cax = ax.imshow(final_array, cmap=cmap)
            cbar = fig.colorbar(cax,ax=ax,shrink=0.5,orientation="horizontal",pad=0.01)
            cbar.formatter.set_powerlimits((0,0))
            ax.axis('off')

        if ii == 0:
            rot_array = array
        else:
            final_array = rot_array + array

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("../simulation-results/rings/final_image/circles.png",dpi=300)

signal = final_array
pixel_count = int(73*3)
max_value = np.max(signal)
signal_count = 0
total_count = 0
center_pixel = int(218/2)
geometric_factor = 2.19409

for x in range(pixel_count):
    for y in range(pixel_count):

        relative_x = (x - center_pixel) * pixel_size
        relative_y = (y - center_pixel) * pixel_size

        aa = np.sqrt(relative_x**2 + relative_y**2)

        # find the geometrical theta angle of the pixel
        angle = np.arctan(aa / distance)

        # signal[y,x] > max_value / 4 and 

        if np.rad2deg(angle) < (theta+0.1):# and np.rad2deg(angle) > (theta - 0.4):
            signal_count += 1
            total_count += signal[y,x]


            #plt.gca().add_patch(rect)

#plt.imshow(signal, cmap=cmap)
#plt.colorbar()
#plt.show()

px_factor = signal_count / (pixel_count**2)
print("recorded flux", total_count  / (geometric_factor * px_factor))