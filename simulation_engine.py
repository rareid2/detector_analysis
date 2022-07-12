from config import create_det_config, create_ca_config
from macros import write_angle_beam_macro, write_PAD_macro, write_pt_macro

import os

EPAD_dir = "/home/rileyannereid/workspace/geant4/EPAD_geant4"


class SimulationEngine:
    def __init__(self, sim_type) -> None:
        self.sim_type = sim_type

        # set some params from geant
        self.env_sizeXY = 2600  # cm
        self.env_sizeZ = 1111  # cm
        self.world_sizeXY = 1.1 * self.env_sizeXY
        self.world_sizeZ = 1.1 * self.env_sizeZ
        self.world_offset = self.env_sizeZ * 0.45
        self.detector_placement = self.world_offset  # im cm

        cwd = os.getcwd()
        self.cwd = cwd

        os.chdir(EPAD_dir)
        try:
            os.system("git checkout %s" % self.sim_type)
        except:
            raise ValueError("enter a valid branch name")

        # build the code for the desired configuration
        os.chdir("build")
        os.system("cmake .. & make")
        os.chdir("..")
        os.chdir(self.cwd)

    def det_config(self, det1_thickness_um=140, det_gap_mm=30, win_thickness_um=100):
        self.det1_thickness_um = det1_thickness_um
        self.det_gap_mm = det_gap_mm
        self.win_thickness_um = win_thickness_um

    def ca_config(
        self,
        n_elements=133,
        mask_thickness_um=400,
        mask_gap_cm=3,
        element_size_mm=0.66,
        mosaic=True,
    ):
        self.n_elements = n_elements
        self.mask_thickness_um = mask_thickness_um
        self.mask_gap_cm = mask_gap_cm
        self.element_size_mm = element_size_mm
        if mosaic:
            # find number in original pattern
            rr = int((n_elements + 1) / 2)
            m_str = "mosaic"
        else:
            rr = n_elements
            m_str = ""

        # define mask size
        ms = round(n_elements * element_size_mm, 2)

        # define aperture filename
        self.aperture_filename = "%d%sMURA_matrix_%.2f.txt" % (
            rr,
            m_str,
            ms,
        )

    def write_config(
        self,
        det1_thickness_um=140,
        det_gap_mm=30,
        win_thickness_um=100,
        n_elements=133,
        mask_thickness_um=400,
        mask_gap_cm=3,
        element_size_mm=0.66,
    ):
        self.det_config()
        create_det_config(
            det1_thickness_um=det1_thickness_um,
            det_gap_mm=det_gap_mm,
            win_thickness_um=win_thickness_um,
        )

        if self.sim_type != "two-detectors":
            self.ca_config()
            create_ca_config(
                n_elements=n_elements,
                mask_thickness_um=mask_thickness_um,
                mask_gap_cm=mask_gap_cm,
                element_size_mm=element_size_mm,
                aperture_filename=self.aperture_filename,
            )

    def write_macro(
        self,
        n_particles,
        energy_keV,
        positions=[[0, 0, -500]],
        directions=[0],
        PAD_run=1,
    ):
        if self.sim_type == "two-detectors":
            macro_file = write_angle_beam_macro(
                n_particles=n_particles,
                energy_keV=energy_keV
            )
            # update energy
            self.energy_keV = energy_keV
        elif self.sim_type == "ca-pt-source":
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

    def run_simulation(self):
        os.chdir(EPAD_dir)
        cmd = "build/main %s/macros/%s" % (EPAD_dir, self.macro_file)
        os.system(cmd)
        os.chdir(self.cwd)

