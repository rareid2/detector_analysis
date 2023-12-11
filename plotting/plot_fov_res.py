from plot_settings import *

element_size_mm = 0.4

# mask size
mask_size = element_size_mm * (67*2-1)

# detector
det_size_mm = element_size_mm * 67

fovs = []
ress = []

dd = np.linspace(0.5,5)
for d in dd:
    f = d *10
    fov = np.arctan(((mask_size * np.sqrt(2) /2) - (det_size_mm * np.sqrt(2)/2)) / f)
    res = 2 * np.arctan(element_size_mm / f)
    fovs.append(np.rad2deg(fov))
    ress.append(np.rad2deg(res))

det_f = dd * 10 /  det_size_mm

fig, ax1 = plt.subplots()

# Plot the first set of data on the first x-axis
color = 'tab:pink'
ax1.set_xlabel('f-number')
ax1.set_ylabel('pitch angle FOV [deg]', color=color)
ax1.plot(det_f, fovs, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True,color=color,linestyle='--', linewidth=0.5)
# Create a second x-axis and plot the second set of data on it
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.plot(det_f, ress, color=color)
ax2.set_ylabel('resolution [deg]', color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.grid(True,color=color,linestyle='--', linewidth=0.5)
plt.savefig("/home/rileyannereid/workspace/geant4/simulation-results/fov_res.png", dpi=500, transparent=True)