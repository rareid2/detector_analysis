import numpy as np
import sys
sys.path.insert(1, "../detector_analysis")
from experiment_engine import ExperimentEngine
from hits import Hits
from deconvolution import Deconvolution

from clustering import find_clusters, remove_clusters
from resample import resample_data
from plot_experiment_data import plot_four_subplots, plot_background_subplots, plot_alignment

from experiment_constants import *

import matplotlib.pyplot as plt
from plot_settings import *

    """plot line plots of deconvolved signal from experiment data to compare if peaks are resolved
    """

two_sources = True
# -------------------- set experiment set up ----------------------
isotope = "Cd109"
frames = 250
exposure_s = 1.0

mask_detector_cm = 5
offset_deg1 = 20
n_files = 25

offset_pairs = [(0,2),(0,2)]
source_detector_cms = [44.66, 99.06]
n_filess = [25,50]
framess = [250,612]

fig = plt.figure()

for ci,(offset_pair, source_detector_cm)in enumerate(zip(offset_pairs, source_detector_cms)):
    offset_deg1 = offset_pair[0]

    resampled_data_list = []
    deconvolved_images = []

    max_energy_keV_spectra = 300
    bin_interval = int(max_energy_keV_spectra // 2)
    bins = np.linspace(
        0, max_energy_keV_spectra + int(max_energy_keV_spectra / bin_interval), bin_interval
    )

    min_energy_keV = 19
    max_energy_keV = 27

    #for version in ["V2", "V3", "V4", "V5"]:

    if source_detector_cm < 50:
        if isotope == "background":
            data_folder = "background-testing/box/"
        elif offset_deg1 == 0:
            data_folder = "fwhm/%dcm/" % round(mask_detector_cm)
            #data_folder = "alignment-testing/%s/" % 'v1'
            #data_folder = "long-exposures/%dpt%ddeg" % (
            #    offset_deg1,
            #    10 * (offset_deg1 % 1),
            #)
        else:
            data_folder = "5cm/%dpt%ddeg" % (
                offset_deg1,
                10 * (offset_deg1 % 1),
            )
    else:
        data_folder = "long-exposures/%dpt%ddeg" % (
        offset_deg1,
        10 * (offset_deg1 % 1),
        ) 

    print("reading data from ", data_folder)

    raw_data = np.zeros((n_pixels, n_pixels))
    cleaned_data_all = np.zeros_like(raw_data)
    background_data = np.zeros_like(raw_data)
    background_spectra = np.zeros(len(bins) - 1)
    signal_spectra = np.zeros(len(bins) - 1)

    my_experiment_engine = ExperimentEngine(
        isotope,
        framess[ci],
        exposure_s,
        source_detector_cm,
        mask_detector_cm,
        data_folder,
        n_filess[ci],
    )
    cc = 0

    # ---------------------- get data -------------------------------
    for i in range(n_filess[ci]):
        my_hits = Hits(
            experiment=True, experiment_engine=my_experiment_engine, file_count=i
        )
        #raw_data += my_hits.detector_hits
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

        #cleaned_data_all += cleaned_data
        hits = np.where(cleaned_data > 0)
        hits_array = np.zeros_like(cleaned_data_all)
        hits_array[hits] = 1
        cleaned_data_all += hits_array

        #background_data += background_clusters
        hits = np.where(background_clusters > 0)
        hits_array = np.zeros_like(background_data)
        hits_array[hits] = 1
        background_data += hits_array

        # get spectra from data

        # shove the larger values into the overflow bin
        # (hack to make an overflow bin - https://stackoverflow.com/questions/26218704/matplotlib-histogram-with-collection-bin-for-high-values)
        background_tracks = [
            max_energy_keV_spectra + int(max_energy_keV_spectra / (2 * bin_interval))
            if i > (max_energy_keV_spectra)
            else i
            for i in background_tracks
        ]
        background_hist_counts, binEdges = np.histogram(background_tracks, bins=bins)
        bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
        background_spectra += background_hist_counts

        signal_tracks = [
            max_energy_keV_spectra + int(max_energy_keV_spectra / (2 * bin_interval))
            if i > (max_energy_keV_spectra)
            else i
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
        offset_deg2s = [offset_pair[1]]
        # offset_deg2s = [1]

        for offset_deg2 in offset_deg2s:

            if offset_deg2 == 0:
                data_folder = "fwhm/%dcm/" % mask_detector_cm
            elif source_detector_cm < 50:
                data_folder = "%dcm/%dpt%ddeg" % (
                    mask_detector_cm,
                    offset_deg2,
                    10 * (offset_deg2 % 1),
                )
            else:
                data_folder = "long-exposures/2pt0deg"
            my_experiment_engine = ExperimentEngine(
                isotope,
                framess[ci],
                exposure_s,
                source_detector_cm,
                mask_detector_cm,
                data_folder,
                n_filess[ci],
            )

            # ---------------------- get data -------------------------------
            for i in range(n_filess[ci]):
                my_hits = Hits(
                    experiment=True, experiment_engine=my_experiment_engine, file_count=i
                )
                #raw_data += my_hits.detector_hits
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

                #cleaned_data_all += cleaned_data
                hits = np.where(cleaned_data > 0)
                hits_array = np.zeros_like(cleaned_data_all)
                hits_array[hits] = 1
                cleaned_data_all += hits_array

                #background_data += background_clusters
                hits = np.where(background_clusters > 0)
                hits_array = np.zeros_like(background_data)
                hits_array[hits] = 1
                background_data += hits_array

                # get spectra from data

                # shove the larger values into the overflow bin
                # (hack to make an overflow bin - https://stackoverflow.com/questions/26218704/matplotlib-histogram-with-collection-bin-for-high-values)
                background_tracks = [
                    max_energy_keV_spectra + int(max_energy_keV_spectra / (2 * bin_interval))
                    if i > (max_energy_keV_spectra)
                    else i
                    for i in background_tracks
                ]
                background_hist_counts, binEdges = np.histogram(background_tracks, bins=bins)
                bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
                background_spectra += background_hist_counts

                signal_tracks = [
                    max_energy_keV_spectra + int(max_energy_keV_spectra / (2 * bin_interval))
                    if i > (max_energy_keV_spectra)
                    else i
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
            data=cleaned_data_all,
        )

        # set up deconvolution
        deconvolver = Deconvolution(experiment_data=resampled_data)
        deconvolver.deconvolve(
            experiment=True,
            downsample=int((n_pixels - 2 * trim) / resample_n_pixels),
            trim=trim,
        )
        resampled_data_list.append(resampled_data)
        deconvolved_images.append(deconvolver.deconvolved_image)

        
        colors = ["#023047","#219EBC","#FFB703","#FB8500","#F15025"]
        plt.plot(deconvolver.signal, color=colors[ci])
        plt.xlabel('pixel #')
        plt.ylabel('signal')
        plot_fname = my_hits.fname[:-6] + "-offset-signal-four"
        plt.ylim([-200,1550])
        plt.hlines(np.max(deconvolver.signal),0,121,linestyles='--',color=colors[ci])
        #plt.hlines(np.max(deconvolver.signal)/2,0,121,linestyles='--',color=colors[ci])

plt.savefig("../experiment_results/%s" % plot_fname,dpi=300,transparent=True)
        