import pandas as pd
import numpy as np


class Hits:
    def __init__(self, fname):
        # filename with hits
        self.fname = fname

        # parse the hits
        self.detector_hits = pd.read_csv(
            self.fname,
            names=["det", "x", "y", "z", "energy"],
            dtype={
                "det": np.int8,
                "x": np.float64,
                "y": np.float64,
                "z": np.float64,
                "energy": np.float64,
            },
            delimiter=",",
            on_bad_lines="skip",
            engine="c",
        )

        self.n_entries = len(self.detector_hits["det"])

        if self.n_entries == 0:
            raise ValueError("No particles hits on any detector!")

    def getBothDetHits(self):
        """
        returns hits from particles that hit both detectors -- only particles that hit both detectors

        :return: dict containing positions of hits on detectors (key Position1 and Position2) and Energies (key Energy1 and Energy2)
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

                except KeyError:
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

        return hits_dict

    def getDetHits(self):
        """
        parses hit file and returns hits from only the first detector

        :return: dict containing positions of hits on detector and energies
        """

        posX = []
        posY = []
        energies = []
        for count, el in enumerate(self.detector_hits["det"]):
            # only get hits on the first detector
            if el == 1:
                xpos = self.detector_hits["x"][count]
                zpos = self.detector_hits["y"][count]
                energy_kev = self.detector_hits["energy"][count]

                # save
                posX.append(xpos)
                posY.append(zpos)
                energies.append(energy_kev)

        hits_pos = [(X, Y) for X, Y in zip(posX, posY)]

        hits_dict = {"Position": hits_pos, "Energy": energies}

        return hits_dict


myhits = Hits("/home/rileyannereid/workspace/geant4/data/hits.csv")
print(myhits.getBothDetHits())
