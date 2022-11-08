import matplotlib.pyplot as plt
import numpy as np

"""
just a quick script to create a plot of fwhm at varying distances between the source 
and the detector to determine near and far field regimes for testing
"""

# set up
n_particles = 1000000
energy_type = "Mono"
energy_level = 500  # keV
ranks = [11,31]
distances = [1,5]
source_distances = np.linspace(-500,10,50)

# plotting set up
fig, ax = plt.subplots()
colors = ['#DB3A34', '#FFC857', '#177E89','#69B578']
ci=0

# simulated two ranks and two distances
for rank in ranks:
    for dist in distances:
        file1 = open("../results/parameter_sweeps/timepix_sim/fwhm_%d_%d_%d_%s_%d.txt"
            % (rank, dist, n_particles, energy_type, energy_level), 'r')
        Lines = file1.readlines()
        Lines = [line.strip('\n') for line in Lines]
        Lines = [float(line) for line in Lines]
        plt.plot(source_distances, Lines, colors[ci], label = 'rank %d, focal length %d cm' % (rank, dist))
        ci+=1

# plotting clean up
plt.gca().invert_xaxis()
plt.legend()
ax.grid()
plt.xlabel('distance between detector and source [cm]')
plt.ylabel('FWHM')

# save it
plt.savefig('plotting/near-far-field-comp.png',dpi=500)
