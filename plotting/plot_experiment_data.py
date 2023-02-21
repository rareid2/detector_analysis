import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from numpy.typing import NDArray

from plotting.plot_settings import *

def plot_four_subplots(
    raw_data: NDArray[np.uint16] = None,
    cleaned_data: NDArray[np.uint16] = None,
    resampled_data: NDArray[np.uint16] = None,
    deconvolved_image: NDArray[np.uint16] = None,
    vmins: list = [0, 0],
    vmaxes: list = [100, 1e5],
    save_name: str = 'plot_subplots'
):

    fig, ax = plt.subplots(2, 2, figsize=(10, 6))
    image = ax[0, 0].imshow(raw_data, cmap=cmap, vmin=vmins[0], vmax=vmaxes[0])
    image = ax[0, 1].imshow(cleaned_data, cmap=cmap, vmin=vmins[0], vmax=vmaxes[0])
    image1 = ax[1, 0].imshow(resampled_data, cmap=cmap, vmin=vmins[0], vmax=vmaxes[0])

    image = ax[1, 1].imshow(deconvolved_image, cmap=cmap, vmin=vmins[1], vmax=vmaxes[1])

    ax[0, 0].set_title("Data")
    ax[0, 1].set_title("Background Removed")
    ax[1, 0].set_title("Resampled")
    ax[1, 1].set_title("Deconvolved")

    ax[1, 1].set_xlabel("pixel #")
    ax[1, 0].set_xlabel("pixel #")
    ax[1, 0].set_ylabel("pixel #")
    ax[0, 0].set_ylabel("pixel #")

    # using padding
    fig.tight_layout()

    # damn colorbar
    fig.subplots_adjust(right=0.55, wspace=0.005, hspace=0.2)
    cbar_ax = fig.add_axes([0.55, 0.09, 0.04, 0.85])
    cbar = fig.colorbar(image1, cax=cbar_ax)
    pos = cbar.ax.get_position()
    cbar.ax.set_aspect("auto")
    ax2 = cbar.ax.twinx()
    ax2.set_ylim([vmins[1], vmaxes[1]])
    pos.x0 += 0.02
    cbar.ax.set_position(pos)
    ax2.set_position(pos)

    cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(6))

    yticks_counts = [
        str(int(float(item.get_text()))) for item in cbar.ax.get_yticklabels()
    ]
    yticks_signal = [
        "{:.1e}".format(int(float(item.get_text()))) for item in ax2.get_yticklabels()
    ]
    yticks_signal[0] = "0"
    cbar.ax.set_yticklabels(yticks_counts, fontsize=10)
    ax2.set_yticklabels(yticks_signal, fontsize=10)

    # plt.text(-1.5, 37,'counts')
    # plt.text(1.3,0,'signal')
    # plt.text(-17,30,'%s' % version, fontsize=16)

    plt.savefig("../experiment_results/%s.png" % save_name, dpi=500, bbox_inches="tight")
    plt.clf()