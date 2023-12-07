# functions to create each type of macro file
# run_PAD.mac (main & two-det-pad)
# run_pt_src.mac (ca-pt-src)
# run_angle_beam.mac (two-detectors)

import os, math
import numpy as np
from typing import List, Any


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
    detector_dim: float = 1.408,
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
        f.write("/run/numberOfThreads 12 \n")
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
            maxtheta = np.rad2deg(
                np.arctan(
                    (detector_dim * np.sqrt(2) / 2)
                    / (detector_placement + np.abs(src[2]))
                )
            )

            f.write("/gps/ang/maxtheta %f deg \n" % maxtheta)

            f.write("/gps/ang/minphi 0 deg \n")
            f.write("/gps/ang/maxphi 360 deg \n")
            f.write("/gps/energy " + str(ene) + " keV \n")

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


def write_surface_macro(
    n_particles: int,
    radius_cm: float,
    ene_type: str,
    ene_min_keV: float,
    ene_max_keV: float,
    macro_directory: str = "/home/rileyannereid/workspace/geant4/EPAD_geant4/macros",
    progress_mod: int = 1000,
    fname_tag: str = "test",
    theta: float = None,
    ca_pos: float = 498.91,
    confine: bool = False,
    world_offset: float = 499.95,
    ring: bool = False,
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

    mf = "run_surface.mac"
    macro_path = os.path.join(macro_directory, mf)

    with open(macro_path, "w") as f:
        f.write("/run/numberOfThreads 12 \n")
        f.write("/run/initialize \n")
        f.write("/control/verbose 0 \n")
        f.write("/run/verbose 0 \n")
        f.write("/event/verbose 0 \n")
        f.write("/tracking/verbose 0 \n")

        f.write("/gps/particle e- \n")
        if ring:
            f.write("/gps/pos/type Plane \n")
            f.write("/gps/pos/shape Square \n")
            f.write("/gps/pos/halfx 4.64 cm \n")
            f.write("/gps/pos/halfy 4.64 cm \n")  # 3.72066
            f.write(f"/gps/pos/centre 0 0 {ca_pos-3} cm \n")
        else:
            f.write("/gps/pos/type Surface \n")
            f.write("/gps/pos/shape Sphere \n")
            f.write(f"/gps/pos/radius {radius_cm} cm \n")
            f.write(f"/gps/pos/centre 0 0 499.95 cm \n")  # move it by just a micron

        # center chosen to align with pinhole (maybe change this?)
        # 499.935 is the front face of detector 1
        # f.write(f"/gps/pos/centre 0 {ca_pos} {world_offset} cm \n")  # - for y align
        # pos = [0, ca_pos, world_offset]
        pos = [0, 0, ca_pos]

        # write the confinement to the volume
        if confine:
            f.write("/gps/pos/confine fovCapPV \n")

        # if using user defined distribution
        if theta is not None:
            f.write("/gps/ang/type user \n")
            f.write("/gps/ang/rot1 -1 0 0 \n")
            f.write("/gps/ang/rot2 0 1 0 \n")
            f.write("/gps/ang/surfnorm false \n")
            f.write("/gps/hist/type theta \n")
            f.write(f"/gps/hist/point {np.deg2rad(theta)} 1 \n")
            """
            f.write(f"/gps/hist/point 0.000000 0.000000 \n")
            f.write(f"/gps/hist/point 0.063466 0.063423 \n")
            f.write(f"/gps/hist/point 0.126933 0.126592 \n")
            f.write(f"/gps/hist/point 0.190399 0.189251 \n")
            f.write(f"/gps/hist/point 0.253866 0.251148 \n")
            f.write(f"/gps/hist/point 0.317332 0.312033 \n")
            f.write(f"/gps/hist/point 0.380799 0.371662 \n")
            f.write(f"/gps/hist/point 0.444265 0.429794 \n")
            f.write(f"/gps/hist/point 0.507732 0.486196 \n")
            f.write(f"/gps/hist/point 0.571198 0.540641 \n")
            f.write(f"/gps/hist/point 0.634665 0.592910 \n")
            f.write(f"/gps/hist/point 0.698131 0.642788 \n")
            f.write(f"/gps/hist/point 0.761598 0.690079 \n")
            f.write(f"/gps/hist/point 0.825065 0.734591 \n")
            f.write(f"/gps/hist/point 0.888531 0.776146 \n")
            f.write(f"/gps/hist/point 0.951998 0.814575 \n")
            f.write(f"/gps/hist/point 1.015464 0.849725 \n")
            f.write(f"/gps/hist/point 1.078931 0.881453 \n")
            f.write(f"/gps/hist/point 1.142397 0.909632 \n")
            f.write(f"/gps/hist/point 1.205864 0.934147 \n")
            f.write(f"/gps/hist/point 1.269330 0.954902 \n")
            f.write(f"/gps/hist/point 1.332797 0.971811 \n")
            f.write(f"/gps/hist/point 1.396263 0.984808 \n")
            f.write(f"/gps/hist/point 1.459730 0.993838 \n")
            f.write(f"/gps/hist/point 1.523196 0.998867 \n")
            """
        else:  # doing sphere
            f.write("/gps/ang/type cos \n")
            f.write("/gps/ang/rot1 -1 0 0 \n")
            f.write("/gps/ang/rot2 0 1 0 \n")
            f.write("/gps/ang/mintheta 0 deg \n")
            f.write("/gps/ang/maxtheta 90 deg \n")

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


def generate_hist():
    # write the file called
    with open("../EPAD_geant4/src/sine.txt", "w") as file:
        # Calculate the increment for theta
        delta_theta = (math.pi / 2) / 99

        # Iterate from 0 to pi and write each row
        for i in range(100):
            theta = i * delta_theta
            sin_theta = math.sin(theta)
            file.write(f"{theta} {sin_theta}\n")

    file.close()
