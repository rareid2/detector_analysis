import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import numpy as np
# Function to process a chunk of data


def get_det_hits(
        detector_hits,
        remove_secondaries: bool = False,
        second_axis: str = "y",
        energy_level: float = 500,
        energy_bin=None,
    ) -> dict:
        """
        return a dictionary containing hits on front detector

        params:
        returns:
            hits_dict: contains positions of hits for front detector in particle order, also has energies
        """

        posX = []
        posY = []
        energies = []
        vertex = []
        detector_offset = 1111 * 0.45 - (0.03 / 2)  # TODO: make this dynamic

        secondary_e = 0
        secondary_gamma = 0

        for count, el in enumerate(detector_hits["det"]):
            # only get hits on the first detector
            if el == 1 and remove_secondaries != True:
                xpos = detector_hits["x"][count]
                ypos = detector_hits["y"][count]
                zpos = detector_hits["z"][count]
                energy_keV = detector_hits["energy"][count]

                if energy_bin is not None:
                    if energy_bin[0] < energy_keV < energy_bin[1]:
                        pass
                    else:
                        continue
                else:
                    pass

                if second_axis == "z" and ypos == 0.015:
                    posX.append(xpos)
                    posY.append(zpos - detector_offset)
                    energies.append(energy_keV)
                    vertex.append(
                        np.array(
                            [
                                detector_hits["x0"][count],
                                detector_hits["y0"][count],
                                detector_hits["z0"][count],
                            ]
                        )
                    )
                    # add to secondary count -- anything NOT parent
                    if detector_hits["ID"][count] != 0:
                        if detector_hits["name"][count] == "e-":
                            secondary_e += 1
                        else:
                            secondary_gamma += 1
                elif second_axis == "y" and zpos == detector_offset:
                    posX.append(xpos)
                    posY.append(ypos)
                    energies.append(energy_keV)
                    vertex.append(
                        np.array(
                            [
                                detector_hits["x0"][count],
                                detector_hits["y0"][count],
                                detector_hits["z0"][count],
                            ]
                        )
                    )
                    # add to secondary count -- anything NOT parent
                    if detector_hits["ID"][count] != 0:
                        if detector_hits["name"][count] == "e-":
                            secondary_e += 1
                        else:
                            secondary_gamma += 1
                else:
                    pass

            # if checking for secondaries and want to remove them, only process electrons
            elif el == 1 and remove_secondaries:
                if (
                    detector_hits["ID"][count] == 0
                    and detector_hits["name"][count] == "e-"
                ):
                    xpos = detector_hits["x"][count]
                    ypos = detector_hits["y"][count]
                    zpos = detector_hits["z"][count]
                    energy_keV = detector_hits["energy"][count]

                    if energy_bin is not None:
                        if energy_bin[0] < energy_keV < energy_bin[1]:
                            pass
                        else:
                            continue
                    else:
                        if (
                            energy_level - energy_level * 0.05
                            < energy_keV
                            < energy_level * 0.05 + energy_level
                        ):
                            pass
                        else:
                            continue

                    if second_axis == "z" and ypos == 0.015:
                        posX.append(xpos)
                        posY.append(zpos - detector_offset)
                        energies.append(energy_keV)
                        vertex.append(
                            np.array(
                                [
                                    detector_hits["x0"][count],
                                    detector_hits["y0"][count],
                                    detector_hits["z0"][count],
                                ]
                            )
                        )
                    elif second_axis == "y" and zpos == detector_offset:
                        posX.append(xpos)
                        posY.append(ypos)
                        energies.append(energy_keV)
                        vertex.append(
                            np.array(
                                [
                                    detector_hits["x0"][count],
                                    detector_hits["y0"][count],
                                    detector_hits["z0"][count],
                                ]
                            )
                        )
                    else:
                        pass
            else:
                pass

        hits_pos = [(X, Y) for X, Y in zip(posX, posY)]

        hits_dict = {"Position": hits_pos, "Energy": energies, "Vertices": vertex}

        print("processed detector hits")

        return hits_dict, secondary_e, secondary_gamma


def process_csv(file_path,lines=None,start=None):
    if start:
        skiprange = range(1, start + 1)
    else:
        skiprange = None
    det_hits = pd.read_csv(file_path,names=["det","x","y","z","energy","ID","name","x0","y0","z0"],dtype={
                        "det": np.int8,
                        "x": np.float64,
                        "y": np.float64,
                        "z": np.float64,
                        "energy": np.float64,
                        "ID": np.int8,
                        "name": str,
                        "x0": np.float64,
                        "y0": np.float64,
                        "z0": np.float64,
                    },
                    delimiter=",",
                    on_bad_lines="skip",
                    engine="c",
                    nrows=lines,
                    skiprows=skiprange
                )
    hits_dict, _, _ = get_det_hits(det_hits,remove_secondaries=False, second_axis="y", energy_level=500)

    return hits_dict

import os, multiprocessing as mp

# process file function
def processfile(filename, start=0, stop=0):
    if start == 0 and stop == 0:
        hits_dict = process_csv(filename)
    else:
        #print('reading', start, 'to ', stop)
        lines = stop -start
        hits_dict = process_csv(filename,lines,start)

    return hits_dict

if __name__ == "__main__":
    filename = '../simulation-data/rings/59-3.47-13p00-deg-2_1.42E+08_Mono_500_13p00.csv'

    # get file size and set chuck size
    filesize = os.path.getsize(filename)
    split_size = filesize//50

    # determine if it needs to be split
    if filesize > split_size:
        print('gonna split it')

        # create pool, initialize chunk start location (cursor)
        pool = mp.Pool(10)
        cursor = 0
        results = []
        # for every chunk in the file...
        for chunk in range(filesize // split_size):

            # determine where the chunk ends, is it the last one?
            if cursor + split_size > filesize:
                end = filesize
            else:
                end = cursor + split_size

            # add chunk to process pool, save reference to get results
            proc = pool.apply_async(processfile, args=[filename, cursor, end])
            results.append(proc)

            # setup next chunk
            cursor = end

        # close and wait for pool to finish
        pool.close()
        pool.join()

    else:
        print('not threading')
        hits_dict = process_csv(filename)

