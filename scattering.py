from hits import Hits
from simulation_engine import SimulationEngine

import numpy as np
from numpy.typing import NDArray
import scipy.stats
import matplotlib.pyplot as plt


class Scattering:
    def __init__(
        self,
        hits: Hits,
        simulation_engine: SimulationEngine,
        save_fig_directory: str = "../results/two_detector_config/",
    ) -> None:

        self.hits_dict = hits.hits_dict
        self.energy_keV = simulation_engine.energy_keV
        self.det1_thickness_um = simulation_engine.det1_thickness_um
        self.gap_in_cm = simulation_engine.det_gap_mm / 10
        self.save_fig_directory = save_fig_directory

        return

    def get_thetas(self) -> NDArray[np.uint16]:
        """
        get scattering angles in degrees

        params:
        returns:
            thetas_deg: scattering angles in degress from each hit
        """

        # if no hits on both detectors, exit
        if "Position1" not in self.hits_dict:
            print("cant get delta x, hits only on one detector")
            return

        x1 = [po[0] for po in self.hits_dict["Position1"]]
        y1 = [po[1] for po in self.hits_dict["Position1"]]

        x2 = [po[0] for po in self.hits_dict["Position2"]]
        y2 = [po[1] for po in self.hits_dict["Position2"]]

        dy = np.array(y1) - np.array(y2)
        self.dp = dy

        thetas_deg = np.rad2deg(np.arctan2(dy, self.gap_in_cm))

        self.thetas_deg = thetas_deg

        return self.thetas_deg

    def get_sigma(self) -> float:  # -- need to accommadate input angles when not 0
        """
        get sigma (std dev) of the scattering angles in degrees

        params:
        returns:
            sigma_deg: std deviation of scattering angles in deg
        """

        sigma_deg = np.std(self.thetas_deg)

        self.sigma_deg = sigma_deg

        return self.sigma_deg

    def get_theoretical_dist(
        self,
        material: str = "Si",
        charge_nmbr: int = 1,
        rest_mass_MeV: float = 0.511,
    ) -> float:
        """
        get theoretical distribution of scattering in the detector

        params:
            material:              material of detector/scattering medium
            charge_nmbr:           charge number of incoming particles
            rest_mass_MeV:         rest mass in MeV of incomcing particle
        returns:
            sigma_deg_theoretical: std deviation of the theoretical distribution
        """

        if material == "Be":
            X0 = 352.7597513557388 * 10**3  # in um
            Z_det = 4
            self.material = material
        elif material == "Si":
            Z_det = 14
            X0 = 37 * 10**3  # in um
            self.material = material
        else:
            print("choose Be or Si")
            return

        # charge number
        z = charge_nmbr
        # thickness
        x = self.det1_thickness_um

        # print('characteristic length is:', x/X0)
        if x / X0 > 100:
            print(x, self.energy_keV)

        # updated beta_cp (from src code)
        E = self.energy_keV * 0.001  # convert to ME
        invbeta_cp = (E + rest_mass_MeV) / (E**2 + 2 * rest_mass_MeV * E)

        # term 1 and 2 for the distribution -- urban 2006 eqn 28
        fz = 1 - (0.24 / (Z_det * (Z_det + 1)))
        t1 = 13.6 * invbeta_cp  # in MeV
        t2 = (
            z
            * np.sqrt(x / X0)
            * np.sqrt(1 + 0.105 * np.log(x / X0) + 0.0035 * (np.log(x / X0)) ** 2)
            * fz
        )

        # compute theoretical sigma for normal distribution
        sigma_deg_theoretical = np.rad2deg(t1 * t2)

        self.sigma_deg_theoretical = sigma_deg_theoretical

        return self.sigma_deg_theoretical

    def plot_theoretical(self, save_fig: bool = True) -> None:
        """
        plot theoretical distribution of scattering in the detector

        params:
            save_fig: true or false if u want to save
        returns:
        """

        print("creating figure")

        # sample from normal distribution using the theoretical sigma
        n = 10000
        mu = 0
        x = np.linspace(
            mu - 3 * self.sigma_deg_theoretical, mu + 3 * self.sigma_deg_theoretical, n
        )

        fig = plt.figure()
        cc = "#34C0D2"

        plt.plot(
            x,
            scipy.stats.norm.pdf(x, mu, self.sigma_deg_theoretical),
            color=cc,
            label="theoretical scattering dist",
        )
        plt.fill_between(
            x,
            scipy.stats.norm.pdf(x, mu, self.sigma_deg_theoretical),
            color=cc,
            alpha=0.5,
        )

        plt.xlabel("scattering angle [deg]")
        plt.legend()

        if save_fig:
            fig.savefig(
                "%sth_scattering_%dkeV_%dum_%s.png"
                % (
                    self.save_fig_directory,
                    self.energy_keV,
                    self.det1_thickness_um,
                    self.material,
                ),
                transparent=True,
                dpi=500,
            )
        plt.show()
        plt.close()

        return

    def plot_compare_th_sim(self, save_fig: bool = True) -> None:
        """
        plot comparing theoretical and simulated distribution of scattering in the detector

        params:
            save_fig: true or false if u want to save
        returns:
        """

        print("creating figure")

        # sample from normal distribution using the theoretical sigma
        n = 10000
        mu = 0
        x = np.linspace(
            mu - 3 * self.sigma_deg_theoretical, mu + 3 * self.sigma_deg_theoretical, n
        )

        fig = plt.figure()
        cc = ["#34C0D2", "#7F7F7F"]

        plt.plot(
            x,
            scipy.stats.norm.pdf(x, mu, self.sigma_deg_theoretical),
            color=cc[0],
            label="theoretical",
        )
        plt.fill_between(
            x,
            scipy.stats.norm.pdf(x, mu, self.sigma_deg_theoretical),
            color=cc[0],
            alpha=0.5,
        )
        plt.hist(self.thetas_deg, label="geant simulation", color=cc[1], density=True, bins=30)

        plt.xlabel("scattering angle [deg]")
        plt.legend()

        if save_fig:
            fig.savefig(
                "%scompare_th_scattering_%dkeV_%dum_%s.png"
                % (
                    self.save_fig_directory,
                    self.energy_keV,
                    self.det1_thickness_um,
                    self.material,
                ),
                transparent=True,
                dpi=500,
            )
        plt.show()
        plt.close()

        return
