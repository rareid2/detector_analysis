import random
import matplotlib.pyplot as plt
import numpy as np

num_samples = 1000

random_samples = [random.random() for _ in range(num_samples)]

MaxTheta = 90
MinTheta = 0

sintheta = [
    np.sqrt(
        r * (np.sin(MaxTheta) * np.sin(MaxTheta) - np.sin(MinTheta) * np.sin(MinTheta))
        + np.sin(MinTheta) * np.sin(MinTheta)
    )
    for r in random_samples
]

costheta = [
    np.cos(MinTheta) - r * (np.cos(MinTheta) - np.cos(MaxTheta)) for r in random_samples
]

theta_dist = [np.arcsin(st) for st in sintheta]
iso_dist = [np.arccos(st) for st in costheta]

plt.hist(iso_dist, density=True, bins=30, edgecolor="black", alpha=0.7)
plt.title("Histogram of Random Samples")
plt.xlabel("Value")
plt.ylabel("Frequency")
test_theta = np.arange(0, 1.2, 0.1)
plt.plot(test_theta, 1.2 * np.sin(2 * test_theta))
plt.savefig("test.png")

# ----------------------------------------------------
plt.clf()
theta_r = [random.random() for _ in range(num_samples)]
phi_r = [random.random() for _ in range(num_samples)]
theta = np.arccos(1 - 2 * np.array(theta_r))
phi = np.array(phi_r) * 2 * np.pi

Radius = 1
x = Radius * np.sin(theta) * np.cos(phi)
y = Radius * np.sin(theta) * np.sin(phi)
z = Radius * np.cos(theta)

from mpl_toolkits.mplot3d import Axes3D

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot the points on the sphere's surface
ax.scatter(x, y, z, c="b", marker="o", s=0.1)

plt.savefig("sphere.png")
