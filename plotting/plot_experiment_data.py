import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as mtick
from numpy.typing import NDArray

from plotting.plot_settings import *

"""plot data collected in experiment with minipix edu
"""


def plot_four_subplots(
    raw_data: NDArray[np.uint16] = None,
    cleaned_data: NDArray[np.uint16] = None,
    resampled_data: NDArray[np.uint16] = None,
    deconvolved_image: NDArray[np.uint16] = None,
    vmins: list = [0, 0],
    vmaxes: list = [100, 1e5],
    save_name: str = "plot_subplots",
):
    fig, ax = plt.subplots(2, 2, figsize=(10, 6))
    image = ax[0, 0].imshow(raw_data, cmap=cmap, vmin=vmins[0], vmax=vmaxes[0])
    image = ax[0, 1].imshow(cleaned_data, cmap=cmap, vmin=vmins[0], vmax=vmaxes[0])
    image1 = ax[1, 0].imshow(resampled_data, cmap=cmap, vmin=vmins[0], vmax=vmaxes[0])

    image = ax[1, 1].imshow(deconvolved_image, cmap=cmap, norm=LogNorm())

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
    """
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

    cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(3))
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

    plt.text(-1.5, 2050, "counts")
    plt.text(1.75, 2050, "signal")
    """
    # plt.text(-17,30,'%s' % version, fontsize=16)

    # plt.colorbar(image)
    plt.savefig(
        "../experiment_results/%s.png" % save_name, dpi=500, bbox_inches="tight"
    )
    plt.clf()


def plot_background_subplots(
    raw_data: NDArray[np.uint16] = None,
    cleaned_data: NDArray[np.uint16] = None,
    background_hist: NDArray[np.uint16] = None,
    signal_hist: NDArray[np.uint16] = None,
    bincenters: NDArray[np.uint16] = None,
    vmins: list = [0, 0],
    vmaxes: list = [100, None],
    save_name: str = "plot_background_subplots",
    time: float = 7200,
):
    fig = plt.figure()
    fig.set_figheight(6)
    fig.set_figwidth(10)

    ax1 = plt.subplot2grid(shape=(2, 3), loc=(0, 0), colspan=1)
    ax2 = plt.subplot2grid(shape=(2, 3), loc=(1, 0), colspan=1, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot2grid(shape=(2, 3), loc=(0, 1), colspan=2)
    ax4 = plt.subplot2grid(shape=(2, 3), loc=(1, 1), colspan=2, sharex=ax3, sharey=ax3)

    image = ax1.imshow(raw_data, cmap=cmap, vmin=vmins[0], vmax=vmaxes[0])
    image = ax2.imshow(cleaned_data, cmap=cmap, vmin=vmins[0], vmax=vmaxes[0])

    ax3.loglog(bincenters, background_hist / time, "-", c=hex_list[0])
    ax4.plot(bincenters, signal_hist / time, "-", c=hex_list[0])
    ax3.set_xlim([6, 300])
    ax4.set_xlim([6, 300])

    print(sum(signal_hist / time))

    max_counts = 0.01

    ax1.text(-90, 140, "Data", fontsize=12, rotation=90)
    ax2.text(-90, 225, "Background Removed", fontsize=12, rotation=90)

    ax2.set_xlabel("pixel #")
    ax1.set_ylabel("pixel #")
    ax2.set_ylabel("pixel #")

    xlabels = [item.get_text() for item in ax3.get_xticklabels()]
    xlabels[-2] = "%s+" % xlabels[-2]
    ax3.set_xticklabels(xlabels)

    # ax4.text(
    #    np.average(bincenters) / 2,
    #    max_counts / 2,
    #    "%.3f counts/sec" % (np.sum(signal_hist) / (time)),
    # )

    ax3.set_ylim([0.0001, 0.01])
    ax4.set_ylim([0.0001, 0.01])

    ax4.set_xlabel("energy [keV]")
    ax1.get_xaxis().set_visible(False)
    ax3.get_xaxis().set_visible(False)
    ax3.set_ylabel("counts/sec")
    ax4.set_ylabel("counts/sec")

    # using padding
    fig.tight_layout()

    plt.savefig(
        "../experiment_results/%s.png" % save_name, dpi=500, bbox_inches="tight"
    )
    plt.clf()


def plot_alignment(
    resampled_data_list: NDArray[np.uint16] = None,
    deconvolved_images: NDArray[np.uint16] = None,
    vmins: list = [0, 0],
    vmaxes: list = [100, 1e5],
    save_name: str = "plot_alignment",
):
    hex_list = ["#023047", "#219EBC", "#FFB703", "#FB8500", "#F15025"]
    fig = plt.figure()
    fig.set_figheight(3)
    fig.set_figwidth(9)

    ax1 = plt.subplot2grid(shape=(2, 6), loc=(0, 0), colspan=1)
    ax2 = plt.subplot2grid(shape=(2, 6), loc=(1, 0), colspan=1, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot2grid(shape=(2, 6), loc=(0, 1), colspan=1, sharex=ax1, sharey=ax1)
    ax4 = plt.subplot2grid(shape=(2, 6), loc=(1, 1), colspan=1, sharex=ax1, sharey=ax1)
    ax5 = plt.subplot2grid(
        shape=(2, 6), loc=(0, 2), colspan=2, rowspan=2, sharex=ax1, sharey=ax1
    )
    ax6 = plt.subplot2grid(shape=(2, 6), loc=(0, 4), colspan=2, rowspan=2)

    # plot the data
    vcount = 2
    for ax, resampled_data in zip([ax1, ax2, ax3, ax4], resampled_data_list):
        image = ax.imshow(resampled_data, cmap=cmap, vmin=vmins[0], vmax=vmaxes[0])
        ax.set_title("V%d" % vcount)
        ax.hlines(
            np.arange(0, 120, 2),
            np.zeros(60),
            120 * np.ones(60),
            linewidth=0.1,
            color="w",
        )
        ax.vlines(
            np.arange(0, 120, 2),
            np.zeros(60),
            120 * np.ones(60),
            linewidth=0.1,
            color="w",
        )
        vcount += 1

    ax1.set_ylabel("pixel")
    ax2.set_ylabel("pixel")
    ax2.set_xlabel("pixel")
    ax4.set_xlabel("pixel")
    ax1.get_xaxis().set_visible(False)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)

    # get the other data
    import sys

    fpath = "../../detector_analysis"
    sys.path.insert(0, fpath)
    fpath = "../../coded_aperture_mask_designs"
    sys.path.insert(0, fpath)
    from simulation_engine import SimulationEngine
    from hits import Hits
    from deconvolution import Deconvolution

    simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=False)

    # timepix design
    det_size_cm = 1.408
    pixel = 0.055  # mm

    # ---------- coded aperture set up ---------
    # set number of elements
    rank = 11
    pixels_downsample = 2

    element_size = pixel * pixels_downsample
    n_elements = (2 * rank) - 1

    mask_size = element_size * n_elements
    # set edge trim - can't use all pixels to downsample to integer amount
    trim = 7
    mosaic = True

    # thickness of mask
    thickness = 500  # um

    distance = 1.91
    source_distance = -16.72
    energy_level = 22.2  # keV -- use weight average
    n_particles = 1.69e8

    # --------------set up simulation---------------
    # for distance in distances:
    # for si,(source_distance,n_particles) in enumerate(zip(source_distances, n_particles_list)):
    print("RUNNING", n_particles, "PARTICLES")
    simulation_engine.set_config(
        det1_thickness_um=300,
        det_gap_mm=30,  # gap between first and second (unused detector)
        win_thickness_um=100,  # window is not actually in there
        det_size_cm=det_size_cm,
        n_elements=n_elements,
        mask_thickness_um=thickness,
        mask_gap_cm=distance,
        element_size_mm=element_size,
        mosaic=mosaic,
        mask_size=mask_size,
    )

    # --------------set up source---------------

    simulation_engine.set_macro(
        n_particles=n_particles,
        energy_keV=[energy_level],
        positions=[[0, source_distance, 0]],
        directions=[1],
    )

    # write the macro
    file1 = open("disk_source.mac", "w")
    file1.write("/gps/particle gamma \n")
    file1.write("/gps/pos/type Plane \n")
    file1.write("/gps/pos/shape Circle \n")
    file1.write("/gps/pos/centre 0. %.2f 0 cm \n" % source_distance)
    file1.write("/gps/pos/rot1 1 0 0 \n")
    file1.write("/gps/pos/rot2 0 0 -1 \n")
    file1.write("/gps/pos/radius 0.03175 cm \n")
    file1.write("/gps/ang/type cos \n")
    file1.write("/gps/energy %.2f keV \n" % energy_level)
    file1.write("/run/printProgress 1000000 \n")
    file1.write("/run/beamOn %d \n" % n_particles)
    file1.close()

    # --------------set up data naming---------------
    fname = "/home/rileyannereid/workspace/geant4/experiment_geant4/experiment_stl/simulation-results/hits_180000000_16_1.csv"

    myhits = Hits(fname, experiment_geant4=True)
    myhits.get_experiment_geant4_hits()
    # deconvolution steps
    deconvolver = Deconvolution(myhits, simulation_engine)

    _, deconvoled_image, resampled_data, _ = deconvolver.deconvolve(
        plot_deconvolved_heatmap=False,
        plot_raw_heatmap=False,
        downsample=pixels_downsample,
        trim=trim,
    )

    image = ax5.imshow(resampled_data.T, cmap=cmap, vmin=vmins[0], vmax=vmaxes[0])
    ax5.hlines(
        np.arange(0, 120, 2), np.zeros(60), 120 * np.ones(60), linewidth=0.1, color="w"
    )
    ax5.vlines(
        np.arange(0, 120, 2), np.zeros(60), 120 * np.ones(60), linewidth=0.1, color="w"
    )
    ax5.set_title("simulated")

    xlist = np.linspace(0, 120, 121)
    ylist = np.linspace(0, 120, 121)
    X, Y = np.meshgrid(xlist, ylist)
    cp = ax6.contour(
        X,
        Y,
        deconvoled_image / np.max(deconvoled_image),
        levels=[0.85, 1],
        colors="#F15025",
    )

    import matplotlib.patches as mpatches

    contour_colors = ["#0091AC", "#D5573B", "#6F2DBD", "#9EF7F2"]
    contour_colors = ["#023047", "#FFB703", "#219EBC", "#FB8500"]
    versions = ["V2", "V3", "V4", "V5"]
    all_patches = []
    all_patches.append(mpatches.Patch(color="#F15025", label="sim"))
    for i, cc in zip(range(4), contour_colors):
        cp = ax6.contour(
            X,
            Y,
            deconvolved_images[i] / np.amax(deconvolved_images[i]),
            levels=[0.85, 1],
            colors=cc,
        )
        mypatch = mpatches.Patch(color=cc, label=versions[i])
        all_patches.append(mypatch)

    ax6.hlines(60, 0, 100, linewidth=0.1, color="grey", linestyles="--")
    ax6.vlines(60, 0, 100, linewidth=0.1, color="grey", linestyles="--")

    ax6.set_ylim([46, 76])
    ax6.set_xlim([46, 76])
    ax6.legend(handles=all_patches, loc="upper right")

    # using padding
    fig.tight_layout()

    plt.savefig(
        "../experiment_results/%s.png" % save_name, dpi=500, bbox_inches="tight"
    )
    print(save_name)
    plt.clf()
