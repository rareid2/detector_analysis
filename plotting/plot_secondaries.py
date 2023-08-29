import os
import numpy as np
import matplotlib.pyplot as plt
import re

# Define the arrays
rank_array = [11, 31]
thicknesses = [500, 1250, 2500]
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
mask_size_list = [
    round(es * ne, 2) for (es, ne) in zip(element_size_mm_list, n_elements_list)
]  # mask size in mm

det_size_cm = 1.408
distances = np.linspace(0.1 * 2 * det_size_cm, 1 * 2 * det_size_cm, 5)
energy_levels = np.logspace(2, 4, 10)

# Define the function to read the floats from a file
def read_floats_from_file(filename):
    with open(filename, 'r') as file:
        floats = [float(line.strip()) for line in file]
    return floats

# Regular expression pattern to match the filename format
file_pattern = r'(\d+)-(\d+)-(\d+)_.+_(\d+)-hits.txt'
fig, axes = plt.subplots(nrows=len(distances[2:]), ncols=len(rank_array), figsize=(10, 12), sharex=True, sharey=True)

# Data to store thickness and line color
thickness_data = {500: ('500 um mask', '#E0FBFC'), 1250: ('1250 um mask', '#98C1D9'), 2500: ('2500 um mask', '#3D5A80')}


for ri, rank in enumerate(rank_array):
    for thickness in thicknesses:
        for di, start_distance in enumerate(distances[2:]):
            seconary_particles = []
            for energy_level in energy_levels:

                filename = f"{rank}-{thickness}-{di}_1.00E+07_Mono_{round(energy_level)}-hits.txt"
                floats = read_floats_from_file(os.path.join('/home/rileyannereid/workspace/geant4/simulation-results/secondaries', filename))

                first_float, second_float = floats[0], floats[1]
                secondardies = first_float - second_float

                seconary_particles.append(1e3 * secondardies / 1e7)

            # Get the row and column indices for the subplot
            row_index = di
            col_index = ri

            # Plot the second float on the corresponding subplot
            ax = axes[row_index, col_index]
            label, color = thickness_data[thickness]
            if di == 0 and ri == 0:
                ax.plot(energy_levels/1e3, seconary_particles, label=label,color=color)
            else:
                ax.plot(energy_levels/1e3, seconary_particles,color=color)

distance = [d + ((150 + (thickness/2))*1e-4) for d in distances[2:]]
fstop = np.array(distance)/(2*det_size_cm)
print(fstop)
fov = np.rad2deg(
    np.arctan((((26.1 / 10) - det_size_cm) / 2) / np.array(distance))
)

for i in range(2):
    axes[2,i].set_xlabel("e- energy [MeV]")
for i in range(3):
    axes[i,0].set_ylabel(f"{round(2 * fov[i])} deg fov \n gamma per 1000 e-")


# Add a single legend above the subplots
fig.legend(loc='upper center')

# Adjust subplot spacing to make space fo
fig.tight_layout(rect=[0, 0.05, 1, 0.95])

# Show the plot
plt.savefig('/home/rileyannereid/workspace/geant4/simulation-results/secondaries/secondary_radiation.png',dpi = 300)