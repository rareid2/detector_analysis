from config import write_det_config, write_ca_config
from macros import (
    write_angle_beam_macro,
    write_PAD_macro,
    write_pt_macro,
    write_surface_macro,
)
from typing import List, Any
import os

EPAD_dir = "/home/rileyannereid/workspace/geant4/EPAD_geant4"


class SimulationEngine:
    def __init__(
        self,
        construct: str = "CA",
        source: str = "PS",
        write_files: bool = True,
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
        self.detector_placement = self.world_offset  # in cm

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
        mask_size: float = None,
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
        self.mosaic = mosaic
        if mosaic:
            # find number in original pattern
            rr = int((n_elements + 1) / 2)
            m_str = "mosaic"
            self.mura_elements = rr
        else:
            rr = n_elements
            m_str = ""
            self.mura_elements = rr

        # define mask size if not input
        if mask_size == None:
            ms = round(n_elements * element_size_mm, 2)
        else:
            ms = mask_size
        self.mask_size = ms

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
        mosaic: bool = True,
        mask_size: float = None,
        radius_cm: float = None,
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
            mosaic:            whether or not the design is mosaicked
        returns:
        """

        # TODO: clean up this method

        self.det_config(
            det1_thickness_um=det1_thickness_um,
            det_gap_mm=det_gap_mm,
            win_thickness_um=win_thickness_um,
            det_size_cm=det_size_cm,
        )
        if self.write_files:
            write_det_config(
                det1_thickness_um=det1_thickness_um,
                det_gap_mm=det_gap_mm,
                win_thickness_um=win_thickness_um,
                det_size_cm=det_size_cm,
            )

        if self.construct != "TD":
            self.ca_config(
                n_elements,
                mask_thickness_um,
                mask_gap_cm,
                element_size_mm,
                mosaic,
                mask_size,
            )
            if self.write_files:
                write_ca_config(
                    n_elements=n_elements,
                    mask_thickness_um=mask_thickness_um,
                    mask_gap_cm=mask_gap_cm,
                    element_size_mm=element_size_mm,
                    mask_size_mm=self.mask_size,
                    aperture_filename=self.aperture_filename,
                    radius_cm=radius_cm,
                )

        return

    def set_macro(
        self,
        n_particles: int,
        energy_keV,
        positions=[[0, 0, -500]],
        directions=[0],
        PAD_run: int = 1,
        surface: bool = False,
        radius_cm: float = 25,
        progress_mod: int = 1000,
        fname_tag: str = "test",
        confine: bool = False,
        detector_dim: float = 1.408,
        theta: float = None,
        ring: float = False,
    ) -> None:
        """
        create macro file based on simulation type -- see macros.py for function defs

        params:
        returns:
        """
        # assign for later use
        self.n_particles = n_particles
        self.radius_cm = radius_cm

        # we need the coded aperture position to center the sphere
        with open(os.path.join(EPAD_dir, "coded_aperture_position.txt"), "r") as file:
            # Read the first line and ignore it (assuming it's not the number we want).
            file.readline()

            # Read the number from the second line and convert it to a float (or int if needed).
            second_line = file.readline().strip()

            if not second_line:
                ca_pos = -499.95
            else:
                ca_pos = round(float(second_line) / 10, 3)
                # + 0.001  # bump but not sure why - dont need to y align

        if self.write_files:
            if self.construct == "TD" and self.source == "PS":
                macro_file = write_angle_beam_macro(
                    n_particles=self.n_particles, energy_keV=energy_keV
                )
                # update energy
                self.energy_keV = energy_keV
            elif self.construct == "CA" and self.source == "PS" and surface:
                macro_file = write_surface_macro(
                    n_particles=n_particles,
                    radius_cm=self.radius_cm,
                    ene_type=energy_keV[0],
                    ene_min_keV=energy_keV[1],
                    ene_max_keV=energy_keV[2],
                    progress_mod=progress_mod,
                    fname_tag=fname_tag,
                    theta=theta,
                    ca_pos=ca_pos,
                    confine=confine,
                    world_offset=self.world_offset,
                    ring=ring,
                )
            elif self.construct == "CA" and self.source == "PS" and surface == False:
                macro_file = write_pt_macro(
                    n_particles=self.n_particles,
                    positions=positions,
                    rotations=directions,
                    energies_keV=energy_keV,
                    detector_placement=self.detector_placement
                    - (self.det1_thickness_um * 1e-4 / 2),
                    progress_mod=progress_mod,
                    fname_tag=fname_tag,
                    detector_dim=detector_dim,
                )
            else:
                macro_file = write_PAD_macro(
                    n_particles=self.n_particles,
                    PAD_run=PAD_run,
                    folding_energy_keV=energy_keV,
                )
            self.macro_file = macro_file

        # if just two detectors, save energy to be accessed in scattering class
        if self.construct == "TD":
            self.energy_keV = energy_keV

        return

    def rename_hits(self, fname: str = "../simulation-data/hits.csv") -> None:
        """
        rename the output hits file

        params:
            fname: new data file name and location
        returns:
        """

        os.rename("../simulation-data/hits.csv", fname)

        return

    def build_simulation(self):
        """
        build the simulation

        params:
        returns:
        """

        # build the code for the desired configuration
        # os.chdir(GEANT_dir)
        cwd = os.getcwd()
        # TODO fix this
        # os.system(". ./geant4.sh")
        os.chdir(EPAD_dir)
        os.chdir("build")
        os.system("make clean")
        os.system(
            "cmake -DCONSTRUCT=%s -DPARTICLE_SOURCE=%s .. & make -j4"
            % (self.construct, self.source)
        )
        os.chdir(cwd)

        return

    def run_simulation(
        self,
        fname: str = "../simulation-data/hits.csv",
        build: bool = True,
        debug: bool = False,
        rename: bool = True,
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

        if debug:
            cmd = "gdb build/main"
        else:
            cmd = "build/main %s/macros/%s" % (EPAD_dir, self.macro_file)

        os.system(cmd)
        os.chdir(cwd)

        if rename:
            self.rename_hits(fname)

        print("simulation complete")

        return
