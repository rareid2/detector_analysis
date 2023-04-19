import matplotlib.pyplot as plt
import numpy as np

"""
just a quick script to create a plot of fwhm at varying distances between the source 
and the detector to determine near and far field regimes from simulations focal 1 to 5
"""

# set up
n_particles = 1000000
energy_type = "Mono"
energy_level = 500  # keV
ranks = [11]
distances = [1,5]
source_distances = np.linspace(-500,10,50)

# plotting set up
fig, ax = plt.subplots()
colors = ['#DB3A34', '#FFC857', '#177E89','#A9FFCB']
ci=0

# simulated two ranks and two distances
for rank in ranks:
    for dist in distances:
        file1 = open("../simulation-results/parameter-sweeps/timepix-sim-1/fwhm-test/fwhm_%d_%d_%d_%s_%d.txt"
            % (rank, dist, n_particles, energy_type, energy_level), 'r')
        Lines = file1.readlines()
        Lines = [line.strip('\n') for line in Lines]
        Lines = [float(line) for line in Lines]
        plt.plot(np.abs(source_distances), Lines, colors[ci], label = 'rank %d, focal length %d cm' % (rank, dist))
        ci+=1

# plotting clean up

plt.legend()
ax.grid()
plt.xlabel('distance between detector and source [cm]')
plt.ylabel('FWHM')
plt.xlim([0,300])
plt.ylim([1,2])

#plt.gca().invert_xaxis()

# save it
plt.savefig("../simulation-results/parameter-sweeps/timepix-sim-1/fwhm-test/fwhm_comparison.png",dpi=500)


# find the min and max radii expected
earth_r = 6387.1 # km
e_r_min = earth_r * 1000 * np.sqrt(0.01) * (1/38.9)**3
e_r_max = earth_r * 1000 * np.sqrt(7) * (6/38.9)**3
p_r_min = earth_r * 1000 * np.sqrt(0.100) * (1/11.1)**3
p_r_max = earth_r * 1000 * np.sqrt(400) * (6/11.1)**3

print('min gyro radii of electrons is ', e_r_min)
print('max gyro radii of electrons is ', e_r_max)
print('min gyro radii of protons is ', p_r_min)
print('max gyro radii of protons is ', p_r_max)
