import numpy as np

from experiment_engine import ExperimentEngine
from hits import Hits
from deconvolution import Deconvolution

from clustering import find_clusters, remove_clusters
from resample import resample_data
from plotting.plot_experiment_data import plot_four_subplots

from experiment_constants import *

two_sources = True
# -------------------- set experiment set up ----------------------
isotope = "Cd109"
frames = 250
exposure_s = 1.0
source_detector_cm = 44.66
mask_detector_cm = 2
offset_deg1 = 0
n_files = 25

min_energy_keV = 19
max_energy_keV = 27

if offset_deg1 == 0:
    data_folder = "fwhm/%dcm/" % round(mask_detector_cm)
else:
    data_folder = "%dcm/%dpt%ddeg" % (
        mask_detector_cm,
        offset_deg1,
        10 * (offset_deg1 % 1),
    )

print("reading data from ", data_folder)

raw_data = np.zeros((n_pixels, n_pixels))
cleaned_data_all = np.zeros_like(raw_data)
background_data = np.zeros_like(raw_data)

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
        experiment=True, experiment_engine=my_experiment_engine, file_count=i
    )
    raw_data += my_hits.detector_hits

    region, clusters = find_clusters(my_hits)
    cleaned_data, background_clusters = remove_clusters(
        my_hits, region, clusters, min_energy_keV, max_energy_keV
    )

    cleaned_data_all += cleaned_data
    background_data += background_clusters

    print(
        "READ AND CLEANED FILE %d" % i,
        "WITH ",
        len(np.argwhere(cleaned_data > 0)),
        "COUNTS",
    )

if two_sources:
    # -------------------- set experiment set up ----------------------
    offset_deg2s = [2.5, 5, 7.5, 10, 12.5, 15, 17.5]
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
        # data_folder = "long-exposures/2deg"
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
                experiment=True, experiment_engine=my_experiment_engine, file_count=i
            )
            raw_data += my_hits.detector_hits

            region, clusters = find_clusters(my_hits)
            cleaned_data, background_clusters = remove_clusters(
                my_hits, region, clusters, min_energy_keV, max_energy_keV
            )

            cleaned_data_all += cleaned_data
            background_data += background_clusters

            print(
                "READ AND CLEANED FILE %d" % i,
                "WITH ",
                len(np.argwhere(cleaned_data > 0)),
                "COUNTS",
            )

# ---------------------- analyze -------------------------------
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

# ---------------------- finally plot it!----------------------
if two_sources:
    plot_fname = my_hits.fname[:-6] + "%dpt%ddeg-%dpt%ddeg-offset" % (
        offset_deg1,
        10 * (offset_deg1 % 1),
        offset_deg2,
        10 * (offset_deg2 % 1),
    )
    plot_fname = my_hits.fname[:-6] + "all-offset"
    vmaxes = [60, 7e4]
else:
    plot_fname = my_hits.fname[:-6] + "%dpt%ddeg-offset" % (
        offset_deg1,
        10 * (offset_deg1 % 1),
    )
    vmaxes = [30, 3e4]

plot_four_subplots(
    raw_data=raw_data,
    cleaned_data=cleaned_data_all,
    resampled_data=resampled_data,
    deconvolved_image=deconvolver.deconvolved_image,
    save_name=plot_fname,
    vmaxes=vmaxes,
)
