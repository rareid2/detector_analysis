import numpy as np
import matplotlib.pyplot as plt

import cmocean

cmap = cmocean.cm.thermal

plt.figure()
# hits 40.29 at 12 dps
# hits 69.1 at 19
x_data = np.linspace(0, 69.1, 20)

fname = f"/home/rileyannereid/workspace/geant4/simulation-results/23-3-fwhm-egg-big/xy-0.3-hits-egg.txt"
y_data = np.loadtxt(fname)
y_data = y_data / max(y_data)

# NO EGG
fname = f"/home/rileyannereid/workspace/geant4/simulation-results/23-3-fwhm-egg-big/xy-0.3-hits-no-egg.txt"
y_data2 = np.loadtxt(fname)
# Plot the data
plt.scatter(x_data, y_data2 / max(y_data2), color="gray", label="No collimator")
degree = 2
x_fit = np.linspace(0, 16.03, 12)
coefficients = np.polyfit(x_fit, y_data[:12], degree)
print(coefficients)
# Evaluate the polynomial at the x data points
y_fit = np.polyval([0.0011, -0.082, 1], x_fit)
# plt.plot(x_data[:12], y_fit)

plt.scatter(x_data, y_data, color=cmap(0.1), label="With collimator")
plt.axvspan(40.29, 69.8, color="gray", alpha=0.2)
plt.xlim([-1, 69.8])
# Add labels and legend
plt.xlabel("Pitch angle [deg]")
plt.ylabel("Normalized sensitivity")
plt.text(53, 0.9, "PCFOV")
plt.text(20, 0.9, "FCFOV")
plt.legend()
# Show the plot
plt.savefig(
    f"../simulation-results/egg_sensitivity_big.png",
    dpi=500,
    bbox_inches="tight",
    pad_inches=0.02,
)
