import numpy as np
import sys

sys.path.insert(1, "../detector_analysis")
from experiment_engine import ExperimentEngine
from hits import Hits
from deconvolution import Deconvolution

from clustering import find_clusters, remove_clusters
from resample import resample_data
from plotting.plot_experiment_data import (
    plot_four_subplots,
    plot_background_subplots,
    plot_alignment,
)

from experiment_constants import *

"""
create 5 panel plot of alignment attempts for experiment with minipix 
"""

two_sources = False
# -------------------- set experiment set up ----------------------
isotope = "background"
frames = 360  # 120
exposure_s = 1.0
source_detector_cm = 0  # 44.66  # 16.72
mask_detector_cm = 0  # 2.0  # 1.91
offset_deg1 = 0

n_files = 20
# 15

resampled_data_list = []
deconvolved_images = []

max_energy_keV_spectra = 300
bin_interval = int(max_energy_keV_spectra // 2)
bins = np.linspace(
    0, max_energy_keV_spectra + int(max_energy_keV_spectra / bin_interval), bin_interval
)

min_energy_keV = 19
max_energy_keV = 27

# for version in ["V2", "V3", "V4", "V5"]:

if isotope == "background":
    data_folder = "background-testing/box/"
elif offset_deg1 == 0:
    data_folder = "fwhm/%dcm/" % round(mask_detector_cm)
    # data_folder = "alignment-testing/%s/" % version
    # data_folder = "long-exposures/%dpt%ddeg" % (
    #    offset_deg1,
    #    10 * (offset_deg1 % 1),
    # )
else:
    data_folder = "2cm/%dpt%ddeg" % (
        offset_deg1,
        10 * (offset_deg1 % 1),
    )

print("reading data from ", data_folder)

raw_data = np.zeros((n_pixels, n_pixels))
raw_data_uncleaned = np.zeros((n_pixels, n_pixels))
cleaned_data_all = np.zeros_like(raw_data)
background_data = np.zeros_like(raw_data)
background_spectra = np.zeros(len(bins) - 1)
signal_spectra = np.zeros(len(bins) - 1)

my_experiment_engine = ExperimentEngine(
    isotope,
    frames,
    exposure_s,
    source_detector_cm,
    mask_detector_cm,
    data_folder,
    n_files,
)
cc = 0

# ---------------------- get data -------------------------------
for i in range(n_files):
    my_hits = Hits(
        experiment=True, experiment_engine=my_experiment_engine, file_count=i
    )
    # raw_data += my_hits.detector_hits
    raw_data_uncleaned += my_hits.detector_hits
    hits = np.where(my_hits.detector_hits > 0)
    hits_array = np.zeros_like(my_hits.detector_hits)
    hits_array[hits] = 1
    raw_data += hits_array

    region, clusters = find_clusters(my_hits)
    (
        cleaned_data,
        background_clusters,
        background_tracks,
        signal_tracks,
    ) = remove_clusters(my_hits, region, clusters, min_energy_keV, max_energy_keV)

    # cleaned_data_all += cleaned_data
    hits = np.where(cleaned_data > 0)
    hits_array = np.zeros_like(cleaned_data_all)
    hits_array[hits] = 1
    cleaned_data_all += hits_array

    # background_data += background_clusters
    hits = np.where(background_clusters > 0)
    hits_array = np.zeros_like(background_data)
    hits_array[hits] = 1
    background_data += hits_array

    # get spectra from data

    # shove the larger values into the overflow bin
    # (hack to make an overflow bin - https://stackoverflow.com/questions/26218704/matplotlib-histogram-with-collection-bin-for-high-values)
    background_tracks = [
        (
            max_energy_keV_spectra + int(max_energy_keV_spectra / (2 * bin_interval))
            if i > (max_energy_keV_spectra)
            else i
        )
        for i in background_tracks
    ]
    background_hist_counts, binEdges = np.histogram(background_tracks, bins=bins)
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    background_spectra += background_hist_counts

    signal_tracks = [
        (
            max_energy_keV_spectra + int(max_energy_keV_spectra / (2 * bin_interval))
            if i > (max_energy_keV_spectra)
            else i
        )
        for i in signal_tracks
    ]
    signal_hist_counts, binEdges = np.histogram(signal_tracks, bins=bins)
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    signal_spectra += signal_hist_counts

    cc += len(np.argwhere(cleaned_data > 0))

    print(
        "READ AND CLEANED FILE %d" % i,
        "WITH ",
        cc,
        "COUNTS",
    )

if two_sources:
    # -------------------- set experiment set up ----------------------
    offset_deg2s = [5]
    # offset_deg2s = [1]

    for offset_deg2 in offset_deg2s:

        if offset_deg2 == 0:
            data_folder = "fwhm/%dcm/" % mask_detector_cm
        else:
            data_folder = "%dcm/%dpt%ddeg" % (
                mask_detector_cm,
                offset_deg2,
                10 * (offset_deg2 % 1),
            )
            # data_folder = "long-exposures/2pt0deg"
        my_experiment_engine = ExperimentEngine(
            isotope,
            frames,
            exposure_s,
            source_detector_cm,
            mask_detector_cm,
            data_folder,
            n_files,
        )

        # ---------------------- get data -------------------------------
        for i in range(n_files):
            my_hits = Hits(
                experiment=True,
                experiment_engine=my_experiment_engine,
                file_count=i,
            )
            raw_data_uncleaned += my_hits.detector_hits
            hits = np.where(my_hits.detector_hits > 0)
            hits_array = np.zeros_like(my_hits.detector_hits)
            hits_array[hits] = 1
            raw_data += hits_array

            region, clusters = find_clusters(my_hits)
            (
                cleaned_data,
                background_clusters,
                background_tracks,
                signal_tracks,
            ) = remove_clusters(
                my_hits, region, clusters, min_energy_keV, max_energy_keV
            )

            # cleaned_data_all += cleaned_data
            hits = np.where(cleaned_data > 0)
            hits_array = np.zeros_like(cleaned_data_all)
            hits_array[hits] = 1
            cleaned_data_all += hits_array

            # background_data += background_clusters
            hits = np.where(background_clusters > 0)
            hits_array = np.zeros_like(background_data)
            hits_array[hits] = 1
            background_data += hits_array

            # get spectra from data

            # shove the larger values into the overflow bin
            # (hack to make an overflow bin - https://stackoverflow.com/questions/26218704/matplotlib-histogram-with-collection-bin-for-high-values)
            background_tracks = [
                (
                    max_energy_keV_spectra
                    + int(max_energy_keV_spectra / (2 * bin_interval))
                    if i > (max_energy_keV_spectra)
                    else i
                )
                for i in background_tracks
            ]
            background_hist_counts, binEdges = np.histogram(
                background_tracks, bins=bins
            )
            bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
            background_spectra += background_hist_counts

            signal_tracks = [
                (
                    max_energy_keV_spectra
                    + int(max_energy_keV_spectra / (2 * bin_interval))
                    if i > (max_energy_keV_spectra)
                    else i
                )
                for i in signal_tracks
            ]
            signal_hist_counts, binEdges = np.histogram(signal_tracks, bins=bins)
            bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
            signal_spectra += signal_hist_counts

            cc += len(np.argwhere(cleaned_data > 0))

            print(
                "READ AND CLEANED FILE %d" % i,
                "WITH ",
                cc,
                "COUNTS",
            )

# ---------------------- analyze -------------------------------
if isotope != "background":
    trim = 7
    resample_n_pixels = 121

    resampled_data = resample_data(
        n_pixels=n_pixels,
        trim=trim,
        resample_n_pixels=resample_n_pixels,
        data=raw_data,
    )
    # raw_data[(raw_data != 0) & (raw_data != 1)] = 1
    np.savetxt(
        "/home/rileyannereid/workspace/geant4/experiment_results/Cd109-0-background_raw.txt",
        raw_data,
    )
    np.savetxt(
        "/home/rileyannereid/workspace/geant4/experiment_results/Cd109-0-background_cleaned.txt",
        resampled_data,
    )

    # set up deconvolution
    deconvolver = Deconvolution(experiment_data=resampled_data)
    deconvolver.deconvolve(
        experiment=True,
        downsample=int((n_pixels - 2 * trim) / resample_n_pixels),
        trim=trim,
        delta_decoding=False,
        rotate=False,
    )
    np.savetxt(
        "/home/rileyannereid/workspace/geant4/experiment_results/Cd109-0-background_dc.txt",
        deconvolver.deconvolved_image,
    )
    resampled_data_list.append(resampled_data)
    deconvolved_images.append(deconvolver.deconvolved_image)
    import matplotlib.pyplot as plt
    from plotting.plot_settings import *

    """
    plt.clf()
    colors = ["#023047","#219EBC","#FFB703","#FB8500","#F15025"]
    plt.plot(deconvolver.signal, color=colors[3])
    plt.xlabel('pixel #')
    plt.ylabel('signal')
    plot_fname = my_hits.fname[:-6] + "%dpt%ddeg-%dpt%ddeg-offset-signal" % (
        offset_deg1,
        10 * (offset_deg1 % 1),
        offset_deg2,
        10 * (offset_deg2 % 1),
    )
    plt.ylim([-200,1400])
    plt.hlines(np.max(deconvolver.signal),0,121,linestyles='--',color=colors[3])
    plt.hlines(np.max(deconvolver.signal)/2,0,121,linestyles='--',color=colors[3])
    plt.savefig("../experiment_results/%s" % plot_fname,dpi=300,transparent=True)
    """


# ---------------------- finally plot it!----------------------

if isotope == "background":
    plot_fname = my_hits.fname[:-6] + "background"
    vmaxes = [1, None]
    plot_background_subplots(
        raw_data=raw_data_uncleaned,
        cleaned_data=cleaned_data_all,
        background_hist=background_spectra,
        signal_hist=signal_spectra,
        bincenters=bincenters,
        vmaxes=vmaxes,
        save_name=plot_fname,
        time=n_files * exposure_s * frames,
    )

elif two_sources:
    plot_fname = my_hits.fname[:-6] + "%dpt%ddeg-%dpt%ddeg-offset" % (
        offset_deg1,
        10 * (offset_deg1 % 1),
        offset_deg2,
        10 * (offset_deg2 % 1),
    )
    # plot_fname = my_hits.fname[:-6] + "all-offset"
    vmaxes = [3, 2e3]
    plot_four_subplots(
        raw_data=raw_data,
        cleaned_data=cleaned_data_all,
        resampled_data=resampled_data,
        deconvolved_image=deconvolver.deconvolved_image,
        save_name=plot_fname,
        vmaxes=vmaxes,
    )
else:
    # plot_fname = my_hits.fname[:-6] + "%dpt%ddeg-offset" % (
    #    offset_deg1,
    #    10 * (offset_deg1 % 1),
    # )
    plot_fname = my_hits.fname[:-6] + "-alignment"
    vmaxes = [3, 2e3]
    plot_alignment(
        resampled_data_list=resampled_data_list,
        deconvolved_images=deconvolved_images,
        vmaxes=vmaxes,
        save_name=plot_fname,
    )
    # plot_four_subplots(
    #    raw_data=raw_data,
    #    cleaned_data=cleaned_data_all,
    #    resampled_data=resampled_data,
    #    deconvolved_image=deconvolver.deconvolved_image,
    #    save_name=plot_fname,
    #    vmaxes=vmaxes,
    # )
