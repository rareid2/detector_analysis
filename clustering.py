import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy import interpolate
from typing import Tuple

from hits import Hits
from plotting.plot_settings import cmap


def find_clusters(hits: Hits) -> Tuple[list, list]:
    """
    identify and classify clusters in array of data from minipix EDU
    from https://stackoverflow.com/questions/74268822/numpy-getting-clusters-from-a-2d-array-black-white-image-mask

    params:
        hits : inherits from Hits class to parse data from experimetn output form Pixet Software
    returns:
        region : list of indices corresponding to clusters
        clusters : list of containing ... ? not totally sure
    """

    # find locations of where the data is greater than 0
    wpoint = np.where(hits.detector_hits > 0)
    points = set((x, y) for x, y in zip(*wpoint))

    # find nearest neighbors using set of 9 pixels around a pixel
    def generate_neighbours(point):
        neighbours = [
            (1, -1),
            (1, 0),
            (1, 1),
            (0, -1),
            (0, 0),
            (0, 1),
            (-1, -1),
            (-1, 0),
            (-1, 1),
        ]
        for neigh in neighbours:
            yield tuple(map(sum, zip(point, neigh)))

    # find regions of pixels that are touching
    def find_regions(p, points):
        reg = []
        seen = set()

        def dfs(point):
            if point not in seen:
                seen.add(point)
                if point in points:
                    reg.append(point)
                    points.remove(point)
                    for n in generate_neighbours(point):
                        dfs(n)

        dfs(p)
        return reg

    region = []

    while points:
        cur = next(iter(points))
        reg = find_regions(cur, points)
        region.append(reg.copy())

    # identify clusters, sort by largest
    clusters = {idx: area for idx, area in enumerate(map(len, region))}
    clusters = sorted(clusters.items(), key=lambda x: x[1], reverse=True)

    return region, clusters


def remove_clusters(
    hits: Hits,
    region: list,
    clusters: list,
    min_energy_keV: int = 19,
    max_energy_keV: int = 27,
    plot_clusters: bool = False,
) -> Tuple[NDArray[np.uint16], NDArray[np.uint16]]:
    """
    remove clusters and interpolate over them
    from https://stackoverflow.com/questions/37662180/interpolate-missing-values-2d-python

    params:
        hits : inherits from Hits class to parse data from experimetn output form Pixet Software
        region : list of indices corresponding to clusters
        clusters : list of containing ... ? not totally sure
        min_energy_keV : int of energies to filter below
        max_energy_keV : int of energies to filter equal to and above
        plot_clusters : bool to turn plotting on an doff for each step
    returns:
        cleaned_data : array with clusters removed and interpolated over
        background_clusters : array containing the clusters that were removed
    """

    # create empty frame to be filled with coutns for every hit
    counts_frame = np.zeros_like(hits.detector_hits)

    track_lengths = []
    track_total_energy = []
    for idx, cluster in enumerate(clusters):
        track_length = 0
        energy_sum = 0
        for x, y in region[cluster[0]]:
            counts_frame[x, y] = 1

            # total length in pixels and total energy depositied in keV
            track_length += 1
            energy_sum += hits.detector_hits[x, y]

        track_lengths.append(track_length)
        track_total_energy.append(energy_sum)
    # counts frame should be identical to original data but replaced with 1s for hits instead of energy deposited

    # convert to arrays
    track_lengths = np.array(track_lengths)
    track_total_energy = np.array(track_total_energy)

    # remove clusters based on total energy deposited
    # TODO incorporate more criteria here

    # TODO remove equals sign in below
    clusters_remove = []
    total_energy_below = np.where(track_total_energy <= min_energy_keV)
    total_energy_above = np.where(track_total_energy >= max_energy_keV)
    clusters_remove.extend(total_energy_below[0])
    clusters_remove.extend(total_energy_above[0])

    # convert image with counts to a float
    clusters_removed = counts_frame.astype(float)

    # go through clusters to remove
    for ii in clusters_remove:
        cluster = clusters[ii]
        for x, y in region[cluster[0]]:
            # replace them with NaNs
            clusters_removed[x, y] = np.nan

    nan_inds = np.argwhere(np.isnan(clusters_removed))

    # replace nans with data - interpolate
    x_meshgrid = np.arange(0, clusters_removed.shape[1])
    y_meshgrid = np.arange(0, clusters_removed.shape[0])

    # mask invalid values
    clusters_removed = np.ma.masked_invalid(clusters_removed)
    xx, yy = np.meshgrid(x_meshgrid, y_meshgrid)

    # get only the valid values
    x1 = xx[~clusters_removed.mask]
    y1 = yy[~clusters_removed.mask]
    clusters_interpolated = clusters_removed[~clusters_removed.mask]

    # create interpolation grid
    GD1 = interpolate.griddata(
        (x1, y1), clusters_interpolated.ravel(), (xx, yy), method="linear"
    )

    # now interpolate over removed clusters and create array of ONLY background data
    background_clusters = np.zeros_like(hits.detector_hits)
    cleaned_data = hits.detector_hits

    for nix, niy in nan_inds:
        background_clusters[nix, niy] = hits.detector_hits[nix, niy]
        cleaned_data[nix, niy] = GD1[nix, niy]

    if plot_clusters:
        # plot the clusters with colors showing track length and track total energy deposited
        plot_tracks = np.zeros_like(hits.detector_hits)
        plot_energies = np.zeros_like(hits.detector_hits)
        for idx, (cluster, track_length, track_energy) in enumerate(
            zip(clusters, track_lengths, track_total_energy)
        ):
            for x, y in region[cluster[0]]:
                plot_tracks[x, y] = track_length
                plot_energies[x, y] = track_energy

        image = plt.imshow(counts_frame, cmap=cmap)
        plt.colorbar(image)
        plt.title("identified tracks")
        plt.show()
        plt.clf()

        image = plt.imshow(plot_tracks, cmap=cmap)
        plt.colorbar(image)
        plt.title("track length")
        plt.show()
        plt.clf()

        image = plt.imshow(plot_energies, cmap=cmap)
        plt.colorbar(image)
        plt.title("track total energy")
        plt.show()
        plt.clf()

        image = plt.imshow(clusters_removed, cmap=cmap)
        plt.colorbar(image)
        plt.title("removed tracks")
        plt.show()
        plt.clf()

        image = plt.imshow(background_clusters, cmap=cmap)
        plt.colorbar(image)
        plt.title("identified background")
        plt.show()
        plt.clf()

        image = plt.imshow(cleaned_data, cmap=cmap)
        plt.colorbar(image)
        plt.title("clean data")
        plt.show()
        plt.clf()

    return cleaned_data, background_clusters


# for testing
"""
base_fname = 'Cd109-0005frames-0pt5s-06pt56cm-sd-0pt00cm-md'
# needs abs path to data
myhits = Hits(fname = '/home/rileyannereid/workspace/geant4/experiment-analysis/data/activity-testing/%s/%s-0.txt' % (base_fname, base_fname), experiment=True)
region, clusters = find_clusters(myhits)
remove_clusters(myhits, region, clusters, min_energy_keV = 19, max_energy_keV = 27, plot_clusters=True)
"""
