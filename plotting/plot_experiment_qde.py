import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import interpolate
from plot_settings import *
import sys
fpath = "/home/rileyannereid/workspace/geant4/detector_analysis"
sys.path.insert(0, fpath)
from clustering import find_clusters, remove_clusters
from experiment_engine import ExperimentEngine
from hits import Hits
from estimate_qde import get_activity, estimate_qde, read_qde_curve, estimate_activity, estimate_counts
from experiment_constants import *

    """plot the QDE of the minipix tested with various sources
    """

# ------set up------
n_files = 9
mask_detector_cm = 0.0

co57_activity_bq, co57_activity_bq_unc = get_activity(bq_1_26_2023_co57, d0_co57, half_life_co57, half_life_co57_unc)
ba133_activity_bq, ba133_activity_bq_unc = get_activity(bq_1_26_2023_ba133, d0_ba133, half_life_ba133, half_life_ba133_unc)
cs137_activity_bq, cs137_activity_bq_unc = get_activity(bq_1_30_2023_cs137, d0_cs137, half_life_cs137, half_life_cs137_unc)
eu152_activity_bq, eu152_activity_bq_unc = get_activity(bq_1_30_2023_eu152, d0_eu152, half_life_eu152, half_life_eu152_unc)

energies = []
qdes = []
uncertainties = []

#  -------------- get Co-57 at 6-7keV --------------
isotope = 'Co57'
frames = 50
exposure_s = 0.1
source_detector_cm = 6.56
data_folder = "qde-testing/%s" % isotope
emission_peaks = emission_peaks_co57[np.argwhere(emission_peaks_co57 < 8)]
branching_ratios = branching_ratios_co57[np.argwhere(emission_peaks_co57 < 8)]
branching_total = sum(branching_ratios)
energy_co57_6 = sum([(emp*br/branching_total) for (emp,br) in zip(emission_peaks, branching_ratios)])
min_energy_keV = 4
max_energy_keV = 9

my_experiment_engine = ExperimentEngine(
    isotope,
    frames,
    exposure_s,
    source_detector_cm,
    mask_detector_cm,
    data_folder,
    n_files,
)

counts = 0
total_time = 0

for i in range(n_files):
    my_hits = Hits(
        experiment=True, experiment_engine=my_experiment_engine, file_count=i
    )

    region, clusters = find_clusters(my_hits)
    (
        cleaned_data,
        background_clusters,
        background_tracks,
        signal_tracks,
    ) = remove_clusters(my_hits, region, clusters, min_energy_keV, max_energy_keV)

    hits = np.argwhere(cleaned_data > 0)
    counts += len(hits)
    total_time += frames*exposure_s

qde_co57_6, uncertainty_co57_6 = estimate_qde(counts,total_time,emission_peaks,branching_ratios,source_detector_cm,co57_activity_bq,co57_activity_bq_unc)
energies.append(energy_co57_6)
qdes.append(qde_co57_6)
uncertainties.append(uncertainty_co57_6)

# -------------- get Co-57 at 14.4keV --------------
emission_peaks = emission_peaks_co57[np.argwhere(emission_peaks_co57 > 8)]
branching_ratios = branching_ratios_co57[np.argwhere(emission_peaks_co57 > 8)]
branching_total = sum(branching_ratios)
energy_co57_14 = sum([(emp*br/branching_total) for (emp,br) in zip(emission_peaks, branching_ratios)])
min_energy_keV = 11
max_energy_keV = 16

my_experiment_engine = ExperimentEngine(
    isotope,
    frames,
    exposure_s,
    source_detector_cm,
    mask_detector_cm,
    data_folder,
    n_files,
)

counts = 0
total_time = 0

for i in range(n_files):
    my_hits = Hits(
        experiment=True, experiment_engine=my_experiment_engine, file_count=i
    )

    region, clusters = find_clusters(my_hits)
    (
        cleaned_data,
        background_clusters,
        background_tracks,
        signal_tracks,
    ) = remove_clusters(my_hits, region, clusters, min_energy_keV, max_energy_keV)

    hits = np.argwhere(cleaned_data > 0)
    counts += len(hits)
    total_time += frames*exposure_s

qde_co57_14, uncertainty_co57_14 = estimate_qde(counts,total_time,emission_peaks,branching_ratios,source_detector_cm,co57_activity_bq,co57_activity_bq_unc)

energies.append(energy_co57_14)
qdes.append(qde_co57_14)
uncertainties.append(uncertainty_co57_14)

# -------------- get Ba-133 at 30keV --------------
isotope = "Ba133"
frames = 100
exposure_s = 0.1
source_detector_cm = 9.1
data_folder = "qde-testing/%s" % isotope
emission_peaks = emission_peaks_ba133[np.argwhere(emission_peaks_ba133 < 34)]
branching_ratios = branching_ratios_ba133[np.argwhere(emission_peaks_ba133 < 34)]
branching_total = sum(branching_ratios)
energy_ba133_30 = sum([(emp*br/branching_total) for (emp,br) in zip(emission_peaks, branching_ratios)])
min_energy_keV = 27
max_energy_keV = 32

my_experiment_engine = ExperimentEngine(
    isotope,
    frames,
    exposure_s,
    source_detector_cm,
    mask_detector_cm,
    data_folder,
    n_files,
)

counts = 0
total_time = 0

for i in range(n_files):
    my_hits = Hits(
        experiment=True, experiment_engine=my_experiment_engine, file_count=i
    )

    region, clusters = find_clusters(my_hits)
    (
        cleaned_data,
        background_clusters,
        background_tracks,
        signal_tracks,
    ) = remove_clusters(my_hits, region, clusters, min_energy_keV, max_energy_keV)

    hits = np.argwhere(cleaned_data > 0)
    counts += len(hits)
    total_time += frames*exposure_s

qde_ba133_30, uncertainty_ba133_30 = estimate_qde(counts,total_time,emission_peaks,branching_ratios,source_detector_cm, ba133_activity_bq, ba133_activity_bq_unc)

energies.append(energy_ba133_30)
qdes.append(qde_ba133_30)
uncertainties.append(uncertainty_ba133_30)

# -------------- get Cs-137 at 31keV --------------
isotope = "Cs137"
frames = 50
exposure_s = 0.1
source_detector_cm = 14.18
data_folder = "qde-testing/%s" % isotope
emission_peaks = emission_peaks_cs137[np.argwhere(emission_peaks_cs137 < 34)]
branching_ratios = branching_ratios_cs137[np.argwhere(emission_peaks_cs137 < 34)]
branching_total = sum(branching_ratios)
energy_cs137_31 = sum([(emp*br/branching_total) for (emp,br) in zip(emission_peaks, branching_ratios)])
min_energy_keV = 28
max_energy_keV = 33

my_experiment_engine = ExperimentEngine(
    isotope,
    frames,
    exposure_s,
    source_detector_cm,
    mask_detector_cm,
    data_folder,
    n_files,
)

counts = 0
total_time = 0

for i in range(n_files):
    my_hits = Hits(
        experiment=True, experiment_engine=my_experiment_engine, file_count=i
    )

    region, clusters = find_clusters(my_hits)
    (
        cleaned_data,
        background_clusters,
        background_tracks,
        signal_tracks,
    ) = remove_clusters(my_hits, region, clusters, min_energy_keV, max_energy_keV)

    hits = np.argwhere(cleaned_data > 0)
    counts += len(hits)
    total_time += frames*exposure_s

qde_cs137_31, uncertainty_cs137_31 = estimate_qde(counts,total_time,emission_peaks,branching_ratios,source_detector_cm,cs137_activity_bq, cs137_activity_bq_unc)

energies.append(energy_cs137_31)
qdes.append(qde_cs137_31)
uncertainties.append(uncertainty_cs137_31)

#  -------------- get Ba-133 at 35keV -------------- 
isotope = "Ba133"
frames = 100
exposure_s = 0.1
source_detector_cm = 9.1
data_folder = "qde-testing/%s" % isotope
emission_peaks = emission_peaks_ba133[np.argwhere(emission_peaks_ba133 > 33)]
branching_ratios = branching_ratios_ba133[np.argwhere(emission_peaks_ba133 > 33)]
branching_total = sum(branching_ratios)
energy_ba133_35 = sum([(emp*br/branching_total) for (emp,br) in zip(emission_peaks, branching_ratios)])
min_energy_keV = 32
max_energy_keV = 37

my_experiment_engine = ExperimentEngine(
    isotope,
    frames,
    exposure_s,
    source_detector_cm,
    mask_detector_cm,
    data_folder,
    n_files,
)

counts = 0
total_time = 0

for i in range(n_files):
    my_hits = Hits(
        experiment=True, experiment_engine=my_experiment_engine, file_count=i
    )

    region, clusters = find_clusters(my_hits)
    (
        cleaned_data,
        background_clusters,
        background_tracks,
        signal_tracks,
    ) = remove_clusters(my_hits, region, clusters, min_energy_keV, max_energy_keV)

    hits = np.argwhere(cleaned_data > 0)
    counts += len(hits)
    total_time += frames*exposure_s

qde_ba133_35, uncertainty_ba133_35 = estimate_qde(counts,total_time,emission_peaks,branching_ratios,source_detector_cm, ba133_activity_bq, ba133_activity_bq_unc)

energies.append(energy_ba133_35)
qdes.append(qde_ba133_35)
uncertainties.append(uncertainty_ba133_35)

# -------------- get Cs-137 at 36keV -------------- 
isotope = "Cs137"
frames = 50
exposure_s = 0.1
source_detector_cm = 14.18
data_folder = "qde-testing/%s" % isotope
emission_peaks = emission_peaks_cs137[np.argwhere(emission_peaks_cs137 > 35)]
branching_ratios = branching_ratios_cs137[np.argwhere(emission_peaks_cs137 > 35)]
branching_total = sum(branching_ratios)
energy_cs137_36 = sum([(emp*br/branching_total) for (emp,br) in zip(emission_peaks, branching_ratios)])
min_energy_keV = 33
max_energy_keV = 38

my_experiment_engine = ExperimentEngine(
    isotope,
    frames,
    exposure_s,
    source_detector_cm,
    mask_detector_cm,
    data_folder,
    n_files,
)

counts = 0
total_time = 0

for i in range(n_files):
    my_hits = Hits(
        experiment=True, experiment_engine=my_experiment_engine, file_count=i
    )

    region, clusters = find_clusters(my_hits)
    (
        cleaned_data,
        background_clusters,
        background_tracks,
        signal_tracks,
    ) = remove_clusters(my_hits, region, clusters, min_energy_keV, max_energy_keV)

    hits = np.argwhere(cleaned_data > 0)
    counts += len(hits)
    total_time += frames*exposure_s

qde_cs137_36, uncertainty_cs137_36 = estimate_qde(counts,total_time,emission_peaks,branching_ratios,source_detector_cm, cs137_activity_bq, cs137_activity_bq_unc)

energies.append(energy_cs137_36)
qdes.append(float(qde_cs137_36))
uncertainties.append(uncertainty_cs137_36)

# -------------- get Eu-152 at 39keV -------------- 
isotope = "Eu152"
frames = 50
exposure_s = 0.1
source_detector_cm = 6.56
data_folder = "qde-testing/%s" % isotope
emission_peaks = emission_peaks_eu152[np.argwhere(emission_peaks_eu152 > 35)]
branching_ratios = branching_ratios_eu152[np.argwhere(emission_peaks_eu152 > 35)]
branching_total = sum(branching_ratios)
energy_eu152_39 = sum([(emp*br/branching_total) for (emp,br) in zip(emission_peaks, branching_ratios)])
min_energy_keV = 36
max_energy_keV = 41

my_experiment_engine = ExperimentEngine(
    isotope,
    frames,
    exposure_s,
    source_detector_cm,
    mask_detector_cm,
    data_folder,
    n_files,
)

counts = 0
total_time = 0

for i in range(n_files):
    my_hits = Hits(
        experiment=True, experiment_engine=my_experiment_engine, file_count=i
    )

    region, clusters = find_clusters(my_hits)
    (
        cleaned_data,
        background_clusters,
        background_tracks,
        signal_tracks,
    ) = remove_clusters(my_hits, region, clusters, min_energy_keV, max_energy_keV)

    hits = np.argwhere(cleaned_data > 0)
    counts += len(hits)
    total_time += frames*exposure_s

qde_eu152_39, uncertainty_eu152_39 = estimate_qde(counts,total_time,emission_peaks,branching_ratios,source_detector_cm, eu152_activity_bq, eu152_activity_bq_unc)

energies.append(energy_eu152_39)
qdes.append(float(qde_eu152_39))
uncertainties.append(uncertainty_eu152_39)

# ------------ ------------ PLOTTING ------------ ------------
qde_energies, qde_curve = read_qde_curve('../experiment_results/qde_curve_wpd.txt')

fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(qde_energies, qde_curve, color='#0B132B',label='QDE curve for minipix-edu', zorder=0)
plt.plot(energies, np.ravel(qdes), marker='o', alpha=.8, ms=4, linestyle='', color=hex_list[0],markeredgewidth=.1)

plot_qdes = [6.266, 14.413, 30.27, 31.452, 34.920, 36.304, 39.097]
widths = [7.112-6.266, 0.0005, 30.973-30.27, 32.194-31.452, 35.907-34.920, 37.349-36.304, 40.118-39.097]
colors = ["#008DD5","#008DD5","#169873","#E43F6F","#169873","#E43F6F","#F6F740"]
colors = ["#219EBC","#219EBC","#91F291","#FB8500","#91F291","#FB8500","#F15025"]
for i,(pq,width) in enumerate(zip(plot_qdes,widths)):
    ax.add_patch(Rectangle(xy=(plot_qdes[i], qdes[i]-(uncertainties[i])),width=width, height=2*uncertainties[i], color=colors[i],alpha=0.7, edgecolor=None))

plt.xlim([5,50])
plt.ylim([0,100])

plt.xlabel('incident x-ray energy [keV]')
plt.ylabel('QDE %')
plt.grid()
#plt.legend()
plt.savefig('../experiment_results/QDE-estimates.png',dpi=500)

# ------------ ------------ CD109 ------------ ------------
isotope = 'Cd109'
n_files = 5
exposure_s_s = [0.5, 1, 1, 1, 1]
nframes_s = [5, 5, 10, 15, 30]
distances = [6.56, 11.64, 16.72, 21.8, 26.88]

qde_curve_interp = interpolate.interp1d(qde_energies, qde_curve, fill_value='extrapolate')

min_energy_keV = 19
max_energy_keV = 27

emission_peaks_cd109 = emission_peaks_cd109[:5]
branching_ratios_cd109 = branching_ratios_cd109[:5]
branching_total = sum(branching_ratios_cd109)
energy_cd109 = sum([(emp*br/branching_total) for (emp,br) in zip(emission_peaks_cd109, branching_ratios_cd109)])
qde_cd109 = qde_curve_interp(energy_cd109)
print('QDE ESIMATED FOR CD 109', qde_cd109, ' %')

activities = []
countss = []
uncertainties = []

fnames = ['Cd109-0005frames-0pt5s-06pt56cm-sd-0pt00cm-md', 'Cd109-0005frames-1pt0s-11pt64cm-sd-0pt00cm-md','Cd109-0010frames-1pt0s-16pt72cm-sd-0pt00cm-md','Cd109-0015frames-1pt0s-21pt80cm-sd-0pt00cm-md','Cd109-0030frames-1pt0s-26pt88cm-sd-0pt00cm-md']

for fname, exposure_s, frames, source_detector_cm in zip(fnames, exposure_s_s, nframes_s, distances):
    data_folder = "activity-testing/%s" % fname
    my_experiment_engine = ExperimentEngine(
        isotope,
        frames,
        exposure_s,
        source_detector_cm,
        mask_detector_cm,
        data_folder,
        n_files,
    )
    counts = 0
    total_time = 0

    for i in range(n_files):
        my_hits = Hits(
            experiment=True, experiment_engine=my_experiment_engine, file_count=i
        )

        region, clusters = find_clusters(my_hits)
        (
            cleaned_data,
            background_clusters,
            background_tracks,
            signal_tracks,
        ) = remove_clusters(my_hits, region, clusters, min_energy_keV, max_energy_keV)

        hits = np.argwhere(cleaned_data > 0)
        counts += len(hits)
        total_time += frames*exposure_s
    
    activity_cd109, uncertainty_activity_cd109 = estimate_activity(counts,total_time,emission_peaks_cd109,branching_ratios_cd109,source_detector_cm,qde_cd109/100)
    countss.append(counts/total_time)
    activities.append(activity_cd109)
    uncertainties.append(uncertainty_activity_cd109)

# plot it!
fig = plt.figure()
ax = fig.add_subplot(111)
plt.errorbar(distances,np.array(activities)/37000, yerr=np.array(uncertainties)/37000, color='#0B132B')
plt.ylim([0,3])
plt.xlabel('distance from source to detector [cm]')
plt.ylabel('activity [micro curie]')
plt.savefig('../experiment_results/activity-vs-distance-cd109.png',dpi=500)
plt.clf()

avg_activity = sum(np.array(activities)/37000)/5
print('AVERAGE ACTIVITY', avg_activity, ' MICRO CURIE')
fig = plt.figure()
ax = fig.add_subplot(111)
plt.errorbar(distances,countss, yerr=np.sqrt(np.array(countss)), color='#0B132B')
plt.xlabel('distance from source to detector [cm]')
plt.ylabel('counts')
plt.savefig('../experiment_results/counts-vs-distance-cd109.png',dpi=500)
plt.clf()

counts, unc = estimate_counts(emission_peaks_cd109, branching_ratios_cd109, 45, qde_cd109/100, avg_activity*37000)
print('EXPECTED COUNTS / SEC', counts)