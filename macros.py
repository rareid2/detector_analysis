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
        # f.write("/run/numberOfThreads 40 \n")
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

    return mf


def write_pt_macro(
    n_particles: int,
    positions: List,
    rotations: List,
    energies_keV: List,
    detector_placement: float,
    macro_directory: str = "/home/rileyannereid/workspace/geant4/EPAD_geant4/macros",
    progress_mod: int = 1000,
    fname_tag: str = "test",
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
            f.write("/gps/pos/shape Square \n")
            f.write("/gps/pos/centre " + pos_string + " cm \n")

            # calculate for theta size
            src = np.array(pos)
            detector_loc = np.array([0, 0, detector_placement])
            norm_d = np.linalg.norm(detector_loc - src)
            normal = (detector_loc - src) / norm_d
            # if rotation = 1 calculate direction required to point source at center of detector
            if rot != 0:
                y_ax = np.array([0, 1, 0])

                # found on a reddit forum to get geant to cooperate
                xprime = np.cross(normal, y_ax)
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

            # find max theta
            maxtheta = np.rad2deg(2 * np.arctan((1.408 / 2) / norm_d))
            f.write("/gps/ang/maxtheta %f deg \n" % maxtheta)

            f.write("/gps/ang/minphi 0 deg \n")
            f.write("/gps/ang/maxphi 360 deg \n")
            f.write("/gps/energy " + str(ene) + " keV \n")
            f.write("/analysis/setFileName %s \n" % fname_tag)
            f.write("/analysis/h1/set 1 100 100 1000 keV \n")
            f.write(
                "/analysis/h2/set 1 100 -5 5 cm none linear 100 -5 5 cm none linear \n"
            )
            f.write(
                "/analysis/h2/set 2 100 -5 5 cm none linear 100 -5 5 cm none linear \n"
            )
            f.write(
                "/analysis/h2/set 3 100 -5 5 cm none linear 100 -5 5 cm none linear \n"
            )
            f.write(
                "/analysis/h2/set 4 120 0 360 deg none linear 100 -1 1 none none linear \n"
            )
            f.write(
                "/analysis/h2/set 5 120 0 360 deg none linear  90 0 180 deg none linear \n"
            )
            f.write("/run/printProgress %d \n" % int(progress_mod))
            f.write("/run/beamOn " + str(int(n_particles)) + "\n")

    f.close()
    print("wrote ", macro_path)

    return mf


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


def write_sphere_macro(
    n_particles: int,
    radius_cm: float,
    ene_type: str,
    ene_min_keV: float,
    ene_max_keV: float,
    macro_directory: str = "/home/rileyannereid/workspace/geant4/EPAD_geant4/macros",
    progress_mod: int = 1000,
) -> None:
    """
    create macro file for a point source (or multiple point sources)

    params:
        n_particles:  number of particles to run
        radius_cm: radius of simulation sphere in cm
        ene_type: energy distribution type (see GPS geant)
        ene_min_keV: min energy for distribution type
        ene_max_keV: max energy for distribution type
    returns:
    """

    mf = "run_sphere.mac"
    macro_path = os.path.join(macro_directory, mf)

    with open(macro_path, "w") as f:
        f.write("/run/numberOfThreads 40 \n")
        f.write("/run/initialize \n")
        f.write("/control/verbose 0 \n")
        f.write("/run/verbose 0 \n")
        f.write("/event/verbose 0 \n")
        f.write("/tracking/verbose 0 \n")

        f.write("/gps/particle e- \n")
        f.write("/gps/pos/type Surface \n")
        f.write("/gps/pos/shape Sphere \n")
        f.write("/gps/pos/radius %.2f cm \n" % radius_cm)

        # center chosen to align with pinhole (maybe change this?)
        f.write("/gps/pos/centre 0 0 497.91 cm \n")

        # write the confinement to the volume
        f.write("/gps/pos/confine fovCapPV \n")

        f.write("/gps/ang/type cos \n")
        # f.write("/gps/ang/mintheta 0 deg \n")
        # f.write("/gps/ang/maxtheta 90 deg \n")

        f.write("/gps/ene/type %s \n" % ene_type)
        if ene_type == "Mono":
            f.write("/gps/ene/mono %2.f keV \n" % ene_min_keV)
        else:
            f.write("/gps/ene/min %.2f keV \n" % ene_min_keV)
            f.write("/gps/ene/max %.2f keV \n" % ene_max_keV)

        # get a progress bar
        f.write("/run/printProgress %d \n" % int(progress_mod))
        f.write("/run/beamOn %d \n" % int(n_particles))

    f.close()
    print("wrote ", macro_path)

    return mf
