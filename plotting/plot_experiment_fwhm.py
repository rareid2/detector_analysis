import numpy as np

import sys

sys.path.insert(1, "../detector_analysis")

from experiment_engine import ExperimentEngine
from hits import Hits
from deconvolution import Deconvolution

from clustering import find_clusters, remove_clusters
from resample import resample_data
import matplotlib.pyplot as plt

from experiment_constants import *

from scipy.signal import find_peaks, peak_widths

    """plot fwhm from simulating the experiment and compare to experiment results
    """

# explicit function to normalize array
def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr)) * diff) / diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


# -------------------- set experiment set up ----------------------
isotope = "Cd109"
frames = 612
exposure_s = 1.0
source_detector_cm = 99.06
mask_detector_cm = 5 
# mask_detector_cms = [2, 2.88, 4, 5]

offset_deg1s = [0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5]
offset_deg1s = [0, 1, 2, 3, 4, 5, 6, 7]

offset_deg1s = [0,2.0]

# offset_deg1 = 0

n_files = 50

min_energy_keV = 19
max_energy_keV = 27

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
# colors = ["#04e762", "#f5b700", "#dc0073", "#008bf8", "#89FC00"]
colors = [
    "03071e",
    "370617",
    "6a040f",
    "9d0208",
    "d00000",
    "dc2f02",
    "e85d04",
    "f48c06",
    "faa307",
    "ffba08",
]
colors = ["#%s" % color for color in colors]
cc = 0
# for offset_deg1s in both_offset:
raw_data = np.zeros((n_pixels, n_pixels))
cleaned_data_all = np.zeros_like(raw_data)
background_data = np.zeros_like(raw_data)
for offset_deg1, color in zip(offset_deg1s, colors):

    if offset_deg1 == 0:
        data_folder = "long-exposures/0pt0deg"
    else:
        data_folder = "%dcm/%dpt%ddeg" % (
            mask_detector_cm,
            offset_deg1,
            10 * (offset_deg1 % 1),
        )
        data_folder = "long-exposures/2pt0deg"
    print("reading data from ", data_folder)

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
        (
            cleaned_data,
            background_clusters,
            background_tracks,
            signal_tracks,
        ) = remove_clusters(my_hits, region, clusters, min_energy_keV, max_energy_keV)

        cleaned_data_all += cleaned_data
        background_data += background_clusters

        print(
            "READ AND CLEANED FILE %d" % i,
            "WITH ",
            len(np.argwhere(cleaned_data > 0)),
            "COUNTS",
        )
    cc = +1


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
_,_,fwhm = deconvolver.deconvolve(
    experiment=True,
    downsample=int((n_pixels - 2 * trim) / resample_n_pixels),
    trim=trim,
)
signal = np.sum(deconvolver.deconvolved_image[50:70,:], axis=0)/20
# subtract background
signal = signal - np.average(signal[int(2 * len(signal) // 3) :])
signal = normalize(signal, 0, 450000)
plt.plot(signal, label="%.1f" % offset_deg1, color=color)

    # just find the width of the center peak
    #peaks, _ = find_peaks(signal)
    #results_half = peak_widths(signal, peaks, rel_height=0.5)
    #fwhm = results_half[0]
    #print(max(fwhm))

# ---------------------- finally plot it!----------------------
plot_fname = my_hits.fname[:-6] + "offset-signal"


plot_fname = my_hits.fname[:-6] + "%dpt%ddeg-%dpt%ddeg-offset-signal" % (
    offset_deg1s[0],
    10 * (offset_deg1s[0] % 1),
    offset_deg1s[1],
    10 * (offset_deg1s[1] % 1),
)

plt.legend()
plt.xlabel("pixel #")
plt.ylabel("signal over background")
plt.savefig("../experiment_results/%s.png" % plot_fname, dpi=500, bbox_inches="tight")

# make the line plot here
plt.clf()


