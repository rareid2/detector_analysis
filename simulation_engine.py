from config import write_det_config, write_ca_config
from macros import write_angle_beam_macro, write_PAD_macro, write_pt_macro

import os

EPAD_dir = "/home/rileyannereid/workspace/geant4/EPAD_geant4"

# TODO: commenting and formatting
# TODO: detector size!!!


class SimulationEngine:
    def __init__(self, sim_type, write_files=True) -> None:
        self.sim_type = sim_type
        self.write_files = write_files

        # set some params from geant
        self.env_sizeXY = 2600  # cm
        self.env_sizeZ = 1111  # cm
        self.world_sizeXY = 1.1 * self.env_sizeXY
        self.world_sizeZ = 1.1 * self.env_sizeZ
        self.world_offset = self.env_sizeZ * 0.45
        self.detector_placement = self.world_offset  # im cm

        if self.write_files:
            # switch to desired branch
            cwd = os.getcwd()
            self.cwd = cwd

            os.chdir(EPAD_dir)
            try:
                os.system("git checkout %s" % self.sim_type)
            except:
                raise ValueError("enter a valid branch name")

            os.chdir(self.cwd)

    def det_config(
        self,
        det1_thickness_um=140,
        det_gap_mm=30,
        win_thickness_um=100,
        det_size_cm=6.3,
    ):
        self.det1_thickness_um = det1_thickness_um
        self.det_gap_mm = det_gap_mm
        self.win_thickness_um = win_thickness_um
        self.det_size_cm = det_size_cm

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

    def set_config(
        self,
        det1_thickness_um=140,
        det_gap_mm=30,
        win_thickness_um=100,
        det_size_cm=6.3,
        n_elements=133,
        mask_thickness_um=400,
        mask_gap_cm=3,
        element_size_mm=0.66,
    ):
        self.det_config()
        if self.write_files:
            write_det_config(
                det1_thickness_um=det1_thickness_um,
                det_gap_mm=det_gap_mm,
                win_thickness_um=win_thickness_um,
                det_size_cm=det_size_cm,
            )

        if self.sim_type != "two-detectors":
            self.ca_config()
            if self.write_files:
                write_ca_config(
                    n_elements=n_elements,
                    mask_thickness_um=mask_thickness_um,
                    mask_gap_cm=mask_gap_cm,
                    element_size_mm=element_size_mm,
                    aperture_filename=self.aperture_filename,
                )

    def set_macro(
        self,
        n_particles,
        energy_keV,
        positions=[[0, 0, -500]],
        directions=[0],
        PAD_run=1,
    ):

        if self.write_files:
            if self.sim_type == "two-detectors":
                macro_file = write_angle_beam_macro(
                    n_particles=n_particles, energy_keV=energy_keV
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
        elif self.sim_type == "two-detectors":
            self.energy_keV = energy_keV

    def run_simulation(self):
        # build the code for the desired configuration
        os.chdir(EPAD_dir)
        os.chdir("build")
        os.system("cmake .. & make")
        os.chdir("..")
        cmd = "build/main %s/macros/%s" % (EPAD_dir, self.macro_file)
        os.system(cmd)
        os.chdir(self.cwd)
