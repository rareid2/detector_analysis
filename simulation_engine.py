from config import write_det_config, write_ca_config
from macros import write_angle_beam_macro, write_PAD_macro, write_pt_macro

import os

EPAD_dir = "/home/rileyannereid/workspace/geant4/EPAD_geant4"
GEANT_dir = "/home/rileyannereid/workspace/geant4/geant4.10.07.p02-install/bin/"


class SimulationEngine:
    def __init__(
        self, construct: str = "CA", source: str = "DS", write_files: bool = True
    ) -> None:

        self.construct = construct
        self.source = source
        self.write_files = write_files

        # set some params from geant
        self.env_sizeXY = 2600  # cm
        self.env_sizeZ = 1111  # cm
        self.world_sizeXY = 1.1 * self.env_sizeXY
        self.world_sizeZ = 1.1 * self.env_sizeZ
        self.world_offset = self.env_sizeZ * 0.45
        self.detector_placement = self.world_offset  # im cm

    def det_config(
        self,
        det1_thickness_um: float = 140,
        det_gap_mm: float = 30,
        win_thickness_um: float = 100,
        det_size_cm: float = 6.3,
    ) -> None:
        """
        update detector configuration parameters

        params:
            see self.set_config()
        returns:
        """

        self.det1_thickness_um = det1_thickness_um
        self.det_gap_mm = det_gap_mm
        self.win_thickness_um = win_thickness_um
        self.det_size_cm = det_size_cm

        return

    def ca_config(
        self,
        n_elements: int = 133,
        mask_thickness_um: float = 400,
        mask_gap_cm: float = 3,
        element_size_mm: float = 0.66,
        mosaic: bool = True,
    ) -> None:
        """
        update coded aperture configuration parameters

        params:
            see self.set_config()
        returns:
        """

        self.n_elements = n_elements
        self.mask_thickness_um = mask_thickness_um
        self.mask_gap_cm = mask_gap_cm
        self.element_size_mm = element_size_mm
        if mosaic:
            # find number in original pattern
            rr = int((n_elements + 1) / 2)
            m_str = "mosaic"
            self.mura_elements = rr
        else:
            rr = n_elements
            m_str = ""
            self.mura_elements = rr

        # define mask size
        ms = round(n_elements * element_size_mm, 2)

        # define aperture filename
        self.aperture_filename = "%d%sMURA_matrix_%.2f.txt" % (
            rr,
            m_str,
            ms,
        )

        return

    def set_config(
        self,
        det1_thickness_um: float = 140,
        det_gap_mm: float = 30,
        win_thickness_um: float = 100,
        det_size_cm: float = 6.3,
        n_elements: int = 133,
        mask_thickness_um: float = 400,
        mask_gap_cm: float = 3,
        element_size_mm: float = 0.66,
    ) -> None:
        """
        sets all the config parameters based on sim type and writes them if option selected

        params:
            det1_thickness_um: thickness of front detector in um
            det_gap_mm:        gap between detectors in mm
            win_thickness_um:  thickness of window in um
            det_size_cm:       xy size of front detector in cm
            n_elements:        number of total elements in design
            mask_thickness_um: thickness of coded aperture in um
            mask_gap_cm:       gap between mask and front detector in cm
            element_size_mm:   size in mm of a single element of the design
        returns:
        """

        self.det_config()
        if self.write_files:
            write_det_config(
                det1_thickness_um=det1_thickness_um,
                det_gap_mm=det_gap_mm,
                win_thickness_um=win_thickness_um,
                det_size_cm=det_size_cm,
            )

        if self.construct != "TD":
            self.ca_config()
            if self.write_files:
                write_ca_config(
                    n_elements=n_elements,
                    mask_thickness_um=mask_thickness_um,
                    mask_gap_cm=mask_gap_cm,
                    element_size_mm=element_size_mm,
                    aperture_filename=self.aperture_filename,
                )

        return

    def set_macro(
        self,
        n_particles: int,
        energy_keV,
        positions=[[0, 0, -500]],
        directions=[0],
        PAD_run: int = 1,
    ) -> None:
        """
        create macro file based on simulation type -- see macros.py for function defs

        params:
        returns:
        """

        if self.write_files:
            if self.construct == "TD" and self.source == "PS":
                macro_file = write_angle_beam_macro(
                    n_particles=n_particles, energy_keV=energy_keV
                )
                # update energy
                self.energy_keV = energy_keV
            elif self.construct == "CA" and self.source == "PS":
                macro_file = write_pt_macro(
                    n_particles=n_particles,
                    positions=positions,
                    rotations=directions,
                    energies_keV=energy_keV,
                    world_size=self.world_sizeZ,
                )
            else:
                macro_file = write_PAD_macro(
                    n_particles=n_particles,
                    PAD_run=PAD_run,
                    folding_energy_keV=energy_keV,
                )
            self.macro_file = macro_file

        # if just two detectors, save energy to be accessed in scattering class
        if self.construct == "TD":
            self.energy_keV = energy_keV

        return

    def rename_hits(self, fname: str) -> None:
        """
        rename the output hits file

        params:
            fname: new data file name and location
        returns:
        """

        os.rename("../data/hits.csv", fname)

        return

    def build_simulation(self):
        """
        build the simulation

        params:
        returns:
        """
        print(
            "cmake -DCONSTRUCT=%s -DPARTICLE_SOURCE=%s .. & make"
            % (self.construct, self.source)
        )

        # build the code for the desired configuration
        # os.chdir(GEANT_dir)
        cwd = os.getcwd()
        # TODO fix this
        # os.system(". ./geant4.sh")
        os.chdir(EPAD_dir)
        os.chdir("build")
        os.system("make clean")
        os.system(
            "cmake -DCONSTRUCT=%s -DPARTICLE_SOURCE=%s .. & make"
            % (self.construct, self.source)
        )
        os.chdir(cwd)

        return

    def run_simulation(
        self, fname: str = "../data/hits.csv", build: bool = True
    ) -> None:  # need to add optoin to not rebuild each time (?)
        """
        run macro, rename data file after

        params:
            fname: new data file name and location
            build: build or nah?
        returns:
        """
        if build:
            self.build_simulation()

        cwd = os.getcwd()
        os.chdir(EPAD_dir)
        cmd = "build/main %s/macros/%s" % (EPAD_dir, self.macro_file)
        os.system(cmd)
        os.chdir(cwd)

        self.rename_hits(fname)

        print("simulation complete")

        return
