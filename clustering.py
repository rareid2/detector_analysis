import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.neighbors import NearestNeighbors
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

    # find locations of where the data is greater than 0 - returns x and y array 
    wpoint = np.where(hits.detector_hits > 0)
    # creates a set of x,y coords (unique points only) from the locations of every pixel with more than 0 deposited
    points = set((x, y) for x, y in zip(*wpoint))

    # find nearest neighbors using set of 8 pixels around a pixel - 8 way searching routine
    def generate_neighbours(point):
        neighbours = [
            (1, -1),
            (1, 0),
            (1, 1),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 0),
            (-1, 1),
        ]
        for neigh in neighbours:
            # all this does is creates a tuple that represents the coordinates of the point and its neighbors
            yield tuple(map(sum, zip(point, neigh)))

    # find regions of pixels that are touching
    def find_regions(p, points):
        reg = []
        seen = set()

        def dfs(point):
            # add the point to set, else it has been seen do nothing
            if point not in seen:
                seen.add(point)
                # if the point exists in all points -- will be true for first pass in function, could be false for neighbors
                if point in points:
                    # save the location of it to add to the cluster
                    reg.append(point)
                    # remove so it doesnt get double counted
                    points.remove(point)
                    # return coordinates of the nearest neighbors
                    for n in generate_neighbours(point):
                        # repeat this function for each neighbor
                        dfs(n)

        dfs(p)
        return reg

    region = []

    while points:
        # iterates through the points that have energy > 0
        cur = next(iter(points))
        reg = find_regions(cur, points)
        region.append(reg.copy())

    # identify clusters, sort by largest, area is the number of pixels
    clusters = {idx: area for idx, area in enumerate(map(len, region))}
    clusters = sorted(clusters.items(), key=lambda x: x[1], reverse=True)
    # clusters is the index of the cluster in region and then the length of that region, sorted by length

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
        clusters : list of containing tuple (idx, area) where idx is the idx corresponding to region and area is the total # of pixels in cluster
        min_energy_keV : int of energies to filter below
        max_energy_keV : int of energies to filter equal to and above
        plot_clusters : bool to turn plotting on an doff for each step
    returns:
        cleaned_data : array with clusters removed and interpolated over
        background_clusters : array containing the clusters that were removed
    """

    # create empty frame to be filled with counts for every hit
    counts_frame = np.zeros_like(hits.detector_hits)

    # empty lists to fill
    track_areas = []
    track_total_energy = []
    track_all_energies = []

    # go through all clusters
    for idx, cluster in enumerate(clusters):
        track_area = 0
        energy_sum = 0
        track_energies = []
        for x, y in region[cluster[0]]:
            counts_frame[x, y] = 1

            # total length in pixels and total energy depositied in keV
            track_area += 1
            energy_sum += hits.detector_hits[x, y]

            # energy along each track
            track_energies.append(hits.detector_hits[x, y])

        track_areas.append(track_area)
        track_total_energy.append(energy_sum)
        track_all_energies.append(np.array(track_energies))
    # counts frame should be identical to original data but replaced with 1s for hits instead of energy deposited

    # convert to arrays
    track_areas = np.array(track_areas)
    track_total_energy = np.array(track_total_energy)
    track_all_energies = np.array(track_all_energies, dtype='object')

    # remove clusters based on three criteria: cluster shape, cluster height (max in one pixel), stopping power

    # SHAPE IDENTIFICATION
    track_shapes = track_areas.copy()

    # DOTS are 1 pixel
    dots = np.where(track_areas < 2)
    # replace with 0
    track_shapes[dots] = 0

    # linearity of the track 
    # https://www.tutorialspoint.com/program-to-count-number-of-points-that-lie-on-a-line-in-python
    def linearity(point_crs):
      res = 0
      for i in range(len(point_crs)):
         x1, y1 = point_crs[i][0], point_crs[i][1]
         slopes = {}
         same = 1
         for j in range(i + 1, len(point_crs)):
            x2, y2 = point_crs[j][0], point_crs[j][1]
            if x2 == x1:
               slopes[float("inf")] = slopes.get(float("inf"), 0) + 1
            elif x1 == x2 and y1 == y2:
               same += 1
            else:
               slope = (y2 - y1) / (x2 - x1)
               slopes[slope] = slopes.get(slope, 0) + 1
         if slopes:
            res = max(res, same + max(slopes.values()))
      return res / len(point_crs)

    linearities = np.array([linearity(region[cluster[0]]) for cluster in clusters])

    # SMALL BLOB has between 2 and 4 pixels and linearity is not 1
    small_blobs = np.where((track_shapes > 1) & (track_shapes < 5) & (linearities < 1.0))
    track_shapes[small_blobs] = 0

    # STRAIGHT TRACK (short) has linearity of 1 and between 2 and 4 pixels
    straight_short_tracks =  np.where((track_shapes > 1) & (track_shapes < 5) & (linearities == 1.0))
    track_shapes[straight_short_tracks] = 0
    
    # HEAVY BLOBS have more than 4 pixels and inside pixels (pixels with 4 touching nearest neighbors)
    heavy_blobs = []
    for idx, track in enumerate(track_shapes): 
        if track != 0:
            # use nearest neighbors to distance of each pixel to every otheer pixel
            nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(region[clusters[idx][0]])
            distances, indices = nbrs.kneighbors(region[clusters[idx][0]])
            for distance in distances:
                # find if there is a pixel in the group that is 1 pixel away from 4 other pixels (inside pixel)
                if np.count_nonzero(distance == 1.0) > 3:
                    heavy_blobs.append(idx)
                    # if so, save the index of the group and get out
                    break
    
    track_shapes[heavy_blobs] = 0

    # STRAIGHT TRACK (long) has linearity above 90% and more than 4 pixels and is not a heavy blob
    straight_long_tracks = np.where((track_shapes > 4) & (linearities > 0.9))
    track_shapes[straight_long_tracks] = 0

    # CURLY TRACK has linearity below 90% and more than 4 pixels and is not a heavy blob
    curly_tracks = np.where((track_shapes > 4) & (linearities <= 0.9))
    track_shapes[curly_tracks] = 0

    # anything left over is bad and confusing
    leftover_tracks = np.where(track_shapes > 0)
    if len(leftover_tracks[0] > 0):
        print("ERROR %d LEFTOVER TRACKS CHECK SHAPE CLASSIFICATION ALGORITHM" % len(leftover_tracks[0]))

    # now check HEIGHT (max eneergy in group) and STOPPING POWER
    signal_pixels = [] # these are both x rays / gammas / electrons
    noisy_pixels = []
    only_electrons = []
    protons = []
    ions = []

    rho_silicon = 2.336 # g/cm^3

    # cluster classifcation based on Gohl et al., 2019 -  SATRAM paper using Timepix 300um Si sensor
    for idx, te in enumerate(track_all_energies):
        # noisy pixels are those that are single dots but over 300keV deposited
        if idx in dots[0]:
            if float(te) > 300:
                noisy_pixels.append(idx)
            else: # not over 300 keV, either x ray or electron
                signal_pixels.append(idx)
        # small blobs and straight short tracks are either protons or xrays/electrons depending on energy deposited
        elif idx in small_blobs[0] or idx in straight_short_tracks[0]:
            if len(np.where(te > 300)[0]) > 0:
                protons.append(idx)
            else:
                signal_pixels.append(idx)
        # heavy blobs ions or protons based on stopping power
        elif idx in heavy_blobs: # heavy blobs is formatted as a list
            stopping_power = track_total_energy[idx] / (rho_silicon * track_areas[idx])
            if stopping_power > 100 * 1e3:
                ions.append(idx)
            else:
                protons.append(idx)
        # long tracks are protons or electrons based on stopping power
        elif idx in straight_long_tracks[0]:
            stopping_power = track_total_energy[idx] / (rho_silicon * track_areas[idx])
            if stopping_power > 10 * 1e3:
                protons.append(idx)
            else:
                only_electrons.append(idx)
        # curly tracks are electrons
        elif idx in curly_tracks[0]:
            only_electrons.append(idx)
        else:
            print('something went wrong')


    # plotting! check if classification works:
    for pro in only_electrons: 
        cluster_check = clusters[pro]
        x = [reg[0] for reg in region[cluster_check[0]]]
        y = [reg[1] for reg in region[cluster_check[0]]]

        plt.scatter(x,y,s=0.5,c='blue')
    for pro in protons: 
        cluster_check = clusters[pro]
        x = [reg[0] for reg in region[cluster_check[0]]]
        y = [reg[1] for reg in region[cluster_check[0]]]
        plt.scatter(x,y,s=0.5,c='red')
    signal_check = []
    for pro in signal_pixels: 
        cluster_check = clusters[pro]
        x = [reg[0] for reg in region[cluster_check[0]]]
        y = [reg[1] for reg in region[cluster_check[0]]]
        if 19 <= track_total_energy[pro] <= 27:
            signal_check.append(pro)
            plt.scatter(x,y,s=0.5,c='orange')
        else:
            plt.scatter(x,y,s=0.5,c='green')
    for pro in ions: 
        cluster_check = clusters[pro]
        x = [reg[0] for reg in region[cluster_check[0]]]
        y = [reg[1] for reg in region[cluster_check[0]]]
        plt.scatter(x,y,s=0.5,c='purple')
    for pro in noisy_pixels: 
        cluster_check = clusters[pro]
        x = [reg[0] for reg in region[cluster_check[0]]]
        y = [reg[1] for reg in region[cluster_check[0]]]
        plt.scatter(x,y,s=0.5,c='yellow')
    plt.xlim([0,256])
    plt.ylim([0,256])
    plt.savefig('test.png')
    plt.clf()

    # TODO remove equals sign in below?
    clusters_remove = []
    total_energy_below = np.where(track_total_energy < min_energy_keV)
    total_energy_above = np.where(track_total_energy > max_energy_keV)
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
            zip(clusters, track_areas, track_total_energy)
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
