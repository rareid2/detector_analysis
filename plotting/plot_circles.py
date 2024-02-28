import numpy as np
import matplotlib.pyplot as plt
import cmocean

cmap = cmocean.cm.thermal
import matplotlib.ticker as ticker

def fmt(x, pos):
    if x == 0:
        return "0"
    else:
        a, b = "{:.1e}".format(x).split("e")
        b = int(b)
        a = float(a)
        if a % 1 > 0.1:
            pass
        else:
            a = int(a)
        #print(b)
        return f"{a}"


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
theta = 33.025
# Set the number of rows and columns in the subplot grid
rows, cols = 1, 3

# Create subplots and plot arrays
fig, axes = plt.subplots(rows, cols, figsize=(5.7, 3),sharex=True)

area = 2.445
n_elements_original = 73

def resample(array):
    original_size = len(array)

    new_array = np.zeros((len(array) // 3, len(array) // 3))

    for i in range(0, original_size, 3):
        k = i // 3
        for j in range(0, original_size, 3):
            n = j // 3
            new_array[k, n] = np.sum(array[i : i + 3, j : j + 3])
    array = new_array
    return array 

n_particles = int((3e8* (area * 2) ** 2) * (1 - np.cos(np.deg2rad(theta))))
formatted_theta = "{:.0f}p{:02d}".format(int(theta), int((theta % 1) * 100))

fname_tag = f"{n_elements_original}-{distance}-{formatted_theta}-deg-circle-rotate-0"
fname = f"../simulation-results/rings/final_image/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_dc.txt"
rot_array = np.loadtxt(fname) 
cax = axes[0].imshow(resample(rot_array), cmap=cmap)
axes[0].axis('off')
axes[0].set_title('Mask',fontsize=8,pad=0.02)

cbar = plt.colorbar(cax, ax=axes[0], pad=0.01,orientation='horizontal')
tick_locator = ticker.MaxNLocator(nbins=3)
cbar.locator = tick_locator
cbar.update_ticks()
cbar.ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt))
cbar.ax.tick_params(axis='x', labelsize=8)
#axes[0].text(1.21, -0.005, rf'$\times 10^{int(5)}$', ha='right', va='top', transform=axes[0].transAxes, fontsize=8)
#axes[0].text(0.1, 0.97, "a)", color='white', ha='right', va='top', transform=axes[0].transAxes, fontsize=10)


fname_tag = f"{n_elements_original}-{distance}-{formatted_theta}-deg-circle"
fname = f"../simulation-results/rings/final_image/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_dc.txt"
array = np.loadtxt(fname) 
cax = axes[1].imshow(resample(array), cmap=cmap)
axes[1].axis('off')
axes[1].set_title('Anti-Mask',fontsize=8,pad=0.02)
cbar = plt.colorbar(cax, ax=axes[1], pad=0.01,orientation='horizontal')
tick_locator = ticker.MaxNLocator(nbins=3)
cbar.locator = tick_locator
cbar.update_ticks()
cbar.ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt))
cbar.ax.tick_params(axis='x', labelsize=8)
axes[1].text(0.98, -0.2, rf'$\times 10^{int(5)}$ Reconstructed Signal', ha='right', va='top', transform=axes[1].transAxes, fontsize=8)
#cbar.set_label("Reconstructed Signal",fontsize=8)
#axes[1].text(0.1, 0.97, "b)", color='white', ha='right', va='top', transform=axes[1].transAxes, fontsize=10)


signal = rot_array+array
cax = axes[2].imshow(resample(signal), cmap=cmap)
axes[2].axis('off')
axes[2].set_title('Summed',fontsize=8,pad=0.02)
cbar = plt.colorbar(cax, ax=axes[2], pad=0.01,orientation='horizontal')
tick_locator = ticker.MaxNLocator(nbins=3)
cbar.locator = tick_locator
cbar.update_ticks()
cbar.ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt))
cbar.ax.tick_params(axis='x', labelsize=8)
#axes[2].text(1.21, -0.005, rf'$\times 10^{int(5)}$', ha='right', va='top', transform=axes[2].transAxes, fontsize=8)
#axes[2].text(0.1, 0.97, "c)", color='white', ha='right', va='top', transform=axes[2].transAxes, fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('8_second_order.png', dpi=500,bbox_inches='tight',pad_inches=0.02)