# functions to create each type of macro file
# run_PAD.mac (main & two-det-pad)
# run_pt_src.mac (ca-pt-src)
# run_angle_beam.mac (two-detectors)

import os
import numpy as np
from typing import List


def write_angle_beam_macro(
    n_particles: int,
    energy_keV: float,
    macro_directory: str = "/home/rileyannereid/workspace/geant4/EPAD_geant4/macros",
) -> None:
    """
    create macro file for a particle beam at an angle off the y-axis

    params:
        n_particles: number of particles to run
        energy_keV:  energy to run the beam at in keV
    returns:
    """

    mf = "run_angle_beam.mac"
    macro_path = os.path.join(macro_directory, mf)
    with open(macro_path, "w") as f:
        f.write("/run/numberOfThreads 40 \n")
        f.write("/run/initialize \n")
        f.write("/control/verbose 0 \n")
        f.write("/run/verbose 0 \n")
        f.write("/event/verbose 0 \n")
        f.write("/tracking/verbose 0 \n")

        f.write("/gps/particle e- \n")
        f.write("/gps/position 0 0 -500 cm \n")
        f.write("/gps/pos/type Point \n")
        # direction is toward center of detector
        f.write("/gps/direction 0 0 1 \n")
        f.write("/gps/energy " + str(energy_keV) + " keV \n")

        f.write("/run/beamOn " + str(n_particles) + " \n")
    f.close()
    print("wrote ", macro_path)

    return


def write_pt_macro(
    n_particles: int,
    positions: List,
    rotations: List,
    energies_keV: List,
    world_size: float,
    macro_directory: str = "/home/rileyannereid/workspace/geant4/EPAD_geant4/macros",
) -> None:
    """
    create macro file for a point source (or multiple point sources)

    params:
        n_particles:  number of particles to run
        positions:    list of positions to initalize pt sources
        rotations:    list of whether or not pt source needs rotation 0 = no, 1 = yes
        energies_keV: list of energies to run point soures at in keV
    returns:
    """

    mf = "run_pt_src.mac"
    macro_path = os.path.join(macro_directory, mf)

    with open(macro_path, "w") as f:
        f.write("/run/numberOfThreads 40 \n")
        f.write("/run/initialize \n")
        f.write("/control/verbose 0 \n")
        f.write("/run/verbose 0 \n")
        f.write("/event/verbose 0 \n")
        f.write("/tracking/verbose 0 \n")

        for pos, rot, ene in zip(positions, rotations, energies_keV):
            pos_string = str(pos[0]) + " " + str(pos[1]) + " " + str(pos[2])

            f.write("/gps/particle e- \n")
            f.write("/gps/pos/type Plane \n")
            f.write("/gps/pos/shape Circle \n")
            f.write("/gps/pos/centre " + pos_string + " cm \n")

            # if rotation = 1 calculate direction required to point source at center of detector
            if rot != 0:

                detector_loc = np.array([0, 0, world_size * 0.45])
                src = np.array(pos)
                norm_d = np.linalg.norm(detector_loc - src)
                normal = (detector_loc - src) / norm_d

                z_ax = np.array([0, 0, 1])

                # found on a reddit forum to get geant to cooperate
                xprime = np.cross(normal, z_ax)
                xprime_normd = xprime / np.linalg.norm(xprime)
                yprime = np.cross(normal, xprime_normd)
                yprime_normd = yprime / np.linalg.norm(yprime)

                rot1_string = (
                    str(xprime_normd[0])
                    + " "
                    + str(xprime_normd[1])
                    + " "
                    + str(xprime_normd[2])
                )
                rot2_string = (
                    str(yprime_normd[0])
                    + " "
                    + str(yprime_normd[1])
                    + " "
                    + str(yprime_normd[2])
                )

                f.write("/gps/pos/rot1 " + rot1_string + " \n")
                f.write("/gps/pos/rot2 " + rot2_string + " \n")

            f.write("/gps/ang/type iso \n")
            f.write("/gps/ang/mintheta 0 deg \n")
            f.write("/gps/ang/maxtheta 0.23 deg \n")
            # f.write('/gps/ang/maxtheta 0.08 deg \n')
            f.write("/gps/ang/minphi 0 deg \n")
            f.write("/gps/ang/maxphi 360 deg \n")
            f.write("/gps/energy " + str(ene) + " keV \n")
            f.write("/run/beamOn " + str(n_particles) + "\n")

    f.close()
    print("wrote ", macro_path)

    return


def write_PAD_macro(
    n_particles: int,
    PAD_run: int,
    folding_energy_keV: float,
    macro_directory: str = "/home/rileyannereid/workspace/geant4/EPAD_geant4/macros",
) -> None:
    """
    create macro file for a pitch angle distribution

    params:
        n_particles:        number of particles to run
        PAD_run:            0 = 90 deg, 1 = sin, 2 = sin^2, 3 = triangle
        folding_energy_keV: folding energy of exponential distribution (main energy) in keV
    returns:
    """

    mf = "run_PAD.mac"
    macro_path = os.path.join(macro_directory, mf)

    with open(macro_path, "w") as f:
        f.write("/run/numberOfThreads 40 \n")
        f.write("/run/initialize \n")
        f.write("/control/verbose 0 \n")
        f.write("/run/verbose 0 \n")
        f.write("/event/verbose 0 \n")
        f.write("/tracking/verbose 0 \n")

        f.write("/particleSource/setFoldingEnergy " + str(folding_energy_keV) + " \n")
        # 0 = 90 deg, 1 = sin, 2 = sin^2, 3 = triangle
        f.write("/particleSource/setPitchAngleDistribution " + str(PAD_run) + " \n")
        f.write("/run/beamOn " + str(n_particles) + " \n")
    f.close()
    print("wrote ", macro_path)

    return mf


def find_disp_pos(theta: float, z_disp: float) -> float:
    """
    find displacement position so that the second pt source is a certain angular disp away from first

    params:
        theta:  desired angular displacement in degrees
        z_disp: distance from detector where sources are initialized
    returns:
        x_disp: displacement in x axis for the angular separation of sources
    """

    # find displaced postion needed to get an angular displacement
    x_disp = z_disp * np.tan(np.deg2rad(theta))

    return x_disp
