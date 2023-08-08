import os, re
import pandas as pd
import numpy as np
from experiment_engine import ExperimentEngine
from uncertainty import add_uncertainty


class Hits:
    def __init__(
        self,
        fname: str = None,
        experiment: bool = False,
        experiment_geant4: bool = False,
        experiment_engine: ExperimentEngine = None,
        file_count: int = 0,
    ) -> None:
        # filename containing hits data
        self.fname = fname

        # ---------------------- parse the hits file -----------------------

        # if physical experiment data its already digitized
        # base_filename = os.path.basename(self.fname)
        if experiment:
            # create file name from the distances and # of frames etc.
            # m = re.match(r'(.*)-(.*)frames-(.*)s-(.*)cm-sd-(.*)cm-md', base_filename)
            # experiment_setup = {
            #    "isotope": m.group(1),
            #    "frames": float(m.group(2)),
            #    "exposure_s": float(m.group(3).replace("pt", ".")),
            #    "source_detector_cm": float(m.group(4).replace("pt", ".")),
            #    "mask_detector_cm": float(m.group(5).replace("pt", ".")),
            # }
            fname = "%s-%04dframes-%dpt%ds-%02dpt%02dcm-sd-%dpt%02dcm-md-%d.txt" % (
                experiment_engine.isotope,
                experiment_engine.frames,
                experiment_engine.exposure_s,
                round(10 * (experiment_engine.exposure_s % 1)),
                experiment_engine.source_detector_cm,
                round(100 * (experiment_engine.source_detector_cm % 1)),
                experiment_engine.mask_detector_cm,
                round(100 * (experiment_engine.mask_detector_cm % 1)),
                file_count,
            )
            fname_full_path = os.path.join(experiment_engine.data_folder, fname)
            f = open(fname_full_path, "r")

            # get the array of data in
            lines = [line.split() for line in f]
            lines_array = np.array(
                [
                    [int(line_element.strip("/n")) for line_element in line]
                    for line in lines
                ]
            ).astype(float)
            lines_sum = np.sum(
                np.array([np.sum(line_array) for line_array in lines_array])
            )
            f.close()

            # add to dict
            self.detector_hits = lines_array
            # self.hits_dict = {"data": lines_array}
            self.n_entries = len(np.where(lines_array > 0))
            # add another dictionary with experiment data (distances etc.)
            # self.experiment_setup = experiment_setup

            self.fname = fname

        # geant experiment is set up in a new coordinate system to import CAD files correctly
        elif experiment_geant4:
            self.detector_hits = pd.read_csv(
                self.fname,
                names=["x", "y", "z", "energy"],
                dtype={
                    "x": np.float64,
                    "y": np.float64,
                    "z": np.float64,
                    "energy": np.float64,
                },
                delimiter=",",
                on_bad_lines="skip",
                engine="c",
            )
            self.n_entries = len(self.detector_hits["energy"])

        # general geant4 simulations
        else:
            self.detector_hits = pd.read_csv(
                self.fname,
                names=["det", "x", "y", "z", "energy", "ID", "name"],
                dtype={
                    "det": np.int8,
                    "x": np.float64,
                    "y": np.float64,
                    "z": np.float64,
                    "energy": np.float64,
                    "ID": np.int8,
                    "name": str,
                },
                delimiter=",",
                on_bad_lines="skip",
                engine="c",
            )

            self.n_entries = len(self.detector_hits["det"])

        if self.n_entries == 0:
            raise ValueError("No particles hits on any detector!")

        self.hits_dict = {}

        # get detector size if needed for uncertainty
        # self.det_size_cm = simulation_engine.det_size_cm

        return

    def get_both_det_hits(self) -> dict:
        """
        return a dictionary containing hits on both detectors

        params:
        returns:
            hits_dict: contains positions of hits for each detector in particle order, also has energies
        """

        array_counter = 0

        posX1 = []
        posY1 = []
        energies1 = []

        posX2 = []
        posY2 = []
        energies2 = []

        for count, el in enumerate(self.detector_hits["det"]):
            # pandas series can throw a KeyError if character starts line
            # TODO: replace this with parse command that doesn't import keyerror throwing lines
            while True:
                try:
                    pos1 = self.detector_hits["det"][count]
                    pos2 = self.detector_hits["det"][count + 1]

                    self.detector_hits["x"][count]
                    self.detector_hits["y"][count]

                    self.detector_hits["x"][count + 1]
                    self.detector_hits["y"][count + 1]

                except:
                    count = count + 1
                    if count == self.n_entries:
                        break
                    continue
                break

            # Checks if first hit detector == 1 and second hit detector == 2
            if np.equal(pos1, 1) & np.equal(pos2, 2):
                posX1.append(self.detector_hits["x"][count])
                posY1.append(self.detector_hits["y"][count])

                posX2.append(self.detector_hits["x"][count + 1])
                posY2.append(self.detector_hits["y"][count + 1])

                energies1.append(self.detector_hits["energy"][count])
                energies2.append(self.detector_hits["energy"][count + 1])

                # Successful pair, continues to next possible pair
                count = count + 2
                array_counter = array_counter + 1
            else:
                # Unsuccessful pair, continues
                count = count + 1

        hits_pos1 = [(X1, Y1) for X1, Y1 in zip(posX1, posY1)]
        hits_pos2 = [(X2, Y2) for X2, Y2 in zip(posX2, posY2)]

        hits_dict = {
            "Position1": hits_pos1,
            "Position2": hits_pos2,
            "Energy1": energies1,
            "Energy2": energies2,
        }

        self.hits_dict = hits_dict

        print("processed detector hits")

        return hits_dict

    # get hits for generalized setup in geant4
    def get_det_hits(self, remove_secondaries: bool = False, second_axis: "y") -> dict:
        """
        return a dictionary containing hits on front detector

        params:
        returns:
            hits_dict: contains positions of hits for front detector in particle order, also has energies
        """

        posX = []
        posY = []
        energies = []
        # detector_offset = 1111 * 0.45 - (0.03 / 2)
        for count, el in enumerate(self.detector_hits["det"]):
            # only get hits on the first detector
            if el == 1 and remove_secondaries != True:
                xpos = self.detector_hits["x"][count]
                zpos = self.detector_hits[second_axis][
                    count
                ]  # change to z and use - detector_offset if y align
                energy_kev = self.detector_hits["energy"][count]

                # save
                posX.append(xpos)
                posY.append(zpos)
                energies.append(energy_kev)
            # if checking for secondaries and want to remove them, only process electrons
            elif el == 1 and remove_secondaries:
                if (
                    self.detector_hits["ID"][count] == 0
                    and self.detector_hits["name"][count] == "e-"
                ):
                    xpos = self.detector_hits["x"][count]
                    zpos = self.detector_hits[second_axis][count]
                    energy_kev = self.detector_hits["energy"][count]

                    # save
                    posX.append(xpos)
                    posY.append(zpos)
                    energies.append(energy_kev)
            else:
                pass

        hits_pos = [(X, Y) for X, Y in zip(posX, posY)]

        hits_dict = {"Position": hits_pos, "Energy": energies}

        self.hits_dict = hits_dict

        print("processed detector hits")

        return hits_dict

    # get hits simulated in experiment set up in geant4
    def get_experiment_geant4_hits(self) -> dict:
        """
        return a dictionary containing hits on Minipix EDU from simulated detector in geant4

        params:
        returns:
            hits_dict: contains positions of hits, also has energies
        """

        posX = []
        posY = []
        energies = []
        for count in range(len(self.detector_hits["energy"]) - 1):
            xpos = self.detector_hits["x"][count]
            zpos = self.detector_hits["z"][count]
            energy_kev = self.detector_hits["energy"][count]

            # save
            posX.append(xpos)
            posY.append(zpos)
            energies.append(energy_kev)

        hits_pos = [(X, Y) for X, Y in zip(posX, posY)]

        hits_dict = {"Position": hits_pos, "Energy": energies}

        self.hits_dict = hits_dict

        print("processed detector hits")

        return hits_dict

    def update_pos_uncertainty(
        self, det_size_cm: float, dist_type: str, dist_param: float
    ) -> None:
        """
        return a dictionary containing hits on both detectors

        params:
            dist_type:  "Gaussian", "Poission", or "Uniform" distribution to sample uncertainty from
            dist_param: determined by dist_type, in milimeters
                        if "Gaussian" standard deviation
                        if "Poission" 1/lambda
                        if "Uniform" bounds for uniform
        returns:
        """

        if len(self.hits_dict.keys()) > 2:
            # contains two positions entries
            pos1 = self.hits_dict["Position1"]
            pos2 = self.hits_dict["Position2"]

            unc_pos1 = [
                add_uncertainty(pos, dist_type, dist_param, det_size_cm) for pos in pos1
            ]
            unc_pos2 = [
                add_uncertainty(pos, dist_type, dist_param, det_size_cm) for pos in pos2
            ]

            # update entries
            self.hits_dict["Position1"] = unc_pos1
            self.hits_dict["Position2"] = unc_pos2

        else:
            # contains one position entry
            pos_p = self.hits_dict["Position"]

            unc_pos = [
                add_uncertainty(pos, dist_type, dist_param, self.det_size_cm)
                for pos in pos_p
            ]

            # unpdate entry
            self.hits_dict["Position"] = unc_pos

        return
