# class to process hits and find
from hits import Hits
import numpy as np

# change to get the gap and the thickness and the starting KE from the config


class Scattering:
    def __init__(self, hits: Hits):
        self.hits_dict = hits.hits_dict

    def get_thetas(self, gap_in_cm):
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

        thetas_deg = np.rad2deg(np.arctan2(dy, gap_in_cm))

        self.thetas_deg = thetas_deg

        return self.thetas_deg

    def get_sigma(self):  # -- need to accommadate input angles when not 0
        """
        get the standard deviation of the scattering angles of the particles

        :return:
        """

        sigma_deg = np.std(self.thetas_deg)

        self.sigma_deg = sigma_deg

        return self.sigma_deg

    def get_theoretical_dist(
        self,
        gap_in_cm,
        det1_thickness,
        material="Si",
        charge_nmbr=1,
        rest_mass_MeV=0.511,
    ):
        """
        compute the theoretical distribution associated with scattering particles in silicon

        :return:
        """

        energy_keV = 5000

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
        x = det1_thickness

        # print('characteristic length is:', x/X0)
        if x / X0 > 100:
            print(x, energy_keV)

        # updated beta_cp (from src code)
        E = energy_keV * 0.001  # convert to ME
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

    def plot_theoretical(self, bin_size=1):
        energy_keV = 5000
        det1_thickness = 140
        import plotly.figure_factory as ff
        import plotly.io as pio

        pio.renderers.default = "notebook"

        print("creating figure")

        # sample from normal distribution using the theoretical sigma
        n = 1000
        th_values = np.random.normal(0, self.sigma_deg_theoretical, n)

        # ground data with the actual simulated sigma
        hist_data = [th_values]

        group_labels = ["theoretical"]

        # Create distplot with custom bin_size
        colors = ["rgb(0, 0, 100)"]
        fig = ff.create_distplot(
            hist_data, group_labels, bin_size=bin_size, colors=colors
        )

        fig.update_layout(
            title_text="theoretical distribution <br><sup> energy = %d keV, thickness = %d um of %s </sup>"
            % (energy_keV, det1_thickness, self.material)
        )
        fig.update_xaxes(title_text="scattering angle [deg]")
        # move the legend
        fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))

        fig.write_html(
            "../results/two_detector_config/th_scattering_%dkeV_%dum_%s.html"
            % (energy_keV, det1_thickness, self.material)
        )
        fig.show()

        # small difference because of the window!

    def plot_compare_th_sim(self, bin_size=5):
        energy_keV = 5000
        det1_thickness = 140
        import plotly.figure_factory as ff
        import plotly.io as pio

        pio.renderers.default = "notebook"

        print("creating figure")

        # sample from normal distribution using the theoretical sigma
        # cut off at n to help with plotting issues
        n = 5000
        th_values = np.random.normal(0, self.sigma_deg_theoretical, n)

        # ground data with the actual simulated sigma
        hist_data = [th_values, self.thetas_deg[:n]]

        group_labels = ["theoretical", "geant simulation"]

        # Create distplot with custom bin_size

        colors = ["rgb(0, 0, 100)", "rgb(0, 200, 200)"]
        fig = ff.create_distplot(
            hist_data, group_labels, bin_size=bin_size, colors=colors
        )

        fig.update_layout(
            title_text="comparing theoretical and simulated distributions <br><sup> energy = %d keV, thickness = %d um of %s </sup>"
            % (energy_keV, det1_thickness, self.material)
        )
        fig.update_xaxes(title_text="scattering angle [deg]")
        fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))

        fig.write_html(
            "../results/two_detector_config/th_sim_scattering_%dkeV_%dum_%s.html"
            % (energy_keV, det1_thickness, self.material)
        )
        fig.show()


myhits = Hits("/home/rileyannereid/workspace/geant4/data/hits.csv")
myhits.getBothDetHits()
# myhits.update_pos_uncertainty("Gaussian", 1)
scattering = Scattering(myhits)
scattering.get_thetas(gap_in_cm=3)
scattering.get_theoretical_dist(gap_in_cm=3, det1_thickness=140)
scattering.plot_theoretical(bin_size=1)
scattering.plot_compare_th_sim(bin_size=3)
