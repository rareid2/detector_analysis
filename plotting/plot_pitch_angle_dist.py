import numpy as np
import matplotlib.pyplot as plt


def read_two_column_file(filename):
    column1 = []
    column2 = []

    with open(filename, "r") as file:
        for line in file:
            # Split the line into two columns based on whitespace
            col1, col2 = map(float, line.split())
            column1.append(col1)
            column2.append(col2)
    dataset = np.array([column1, column2])

    return dataset


fname_tag = f"sine-test-d1-try-full-sine-uncorrected"
n_particles = 1e9
energy_type = "Mono"
energy_level = 500

fname = f"../simulation-results/sine-test/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_fov0.txt"
data = read_two_column_file(fname)

plt.plot(data[0], data[1])

# add a sine distribution on here with flat field
n_particles = 1e8
fname_tag = f"sine-test-d1-flat"
fname_flat = f"../simulation-results/sine-test/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_fov.txt"
data_flat = read_two_column_file(fname_flat)

theta_deg = np.linspace(data[0][0], data[0][-1], len(data_flat[1]))
theta_rad = np.deg2rad(theta_deg)

# Compute the probability density function (PDF) of the cosine squared distribution
pdf = (np.sin(theta_rad + np.pi/2))#* np.array(data_flat[1])

# Plot the cosine squared distribution
plt.plot(theta_deg, pdf)

plt.savefig(
    f"../simulation-results/sine-test/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_sine0.png"
)
