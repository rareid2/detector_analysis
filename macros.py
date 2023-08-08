# functions to create each type of macro file
# run_PAD.mac (main & two-det-pad)
# run_pt_src.mac (ca-pt-src)
# run_angle_beam.mac (two-detectors)

import os, math
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
                f"/analysis/h2/set 1 100 {pos[0]-5} {pos[0]+5} cm none linear 100 {pos[1]-5} {pos[1]+5} cm none linear \n"
            )
            f.write(
                f"/analysis/h2/set 2 100 {pos[1]-5} {pos[1]+5} cm none linear 100 {pos[2]-5} {pos[2]+5} cm none linear \n"
            )
            f.write(
                f"/analysis/h2/set 3 100 {pos[2]-5} {pos[2]+5} cm none linear 100 {pos[0]-5} {pos[0]+5} cm none linear \n"
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
    fname_tag: str = "test",
    dist: str = None,
    ca_pos: float = 498.91,
    confine: bool = False,
    world_offset: float = 499.95,
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
        f.write("/run/numberOfThreads 12 \n")
        f.write("/run/initialize \n")
        f.write("/control/verbose 0 \n")
        f.write("/run/verbose 0 \n")
        f.write("/event/verbose 0 \n")
        f.write("/tracking/verbose 0 \n")

        f.write("/gps/particle e- \n")
        f.write("/gps/pos/type Surface \n")
        f.write("/gps/pos/shape Sphere \n")
        f.write(f"/gps/pos/radius {radius_cm} cm \n")

        # center chosen to align with pinhole (maybe change this?)
        # 499.935 is the front face of detector 1
        # f.write(f"/gps/pos/centre 0 {ca_pos} {world_offset} cm \n") #- for y align
        f.write(f"/gps/pos/centre 0 0 {ca_pos} cm \n")
        pos = [0, 0, ca_pos]

        # write the confinement to the volume
        if confine:
            f.write("/gps/pos/confine fovCapPV \n")

        # if using user defined distribution
        if dist:
            f.write("/gps/ang/type user \n")
            # f.write("/gps/ang/rot1 -1 0 0 \n")
            # f.write("/gps/ang/rot2 0 0 1 \n")
            f.write("/gps/ang/surfnorm false \n")
            f.write("/gps/hist/type theta \n")

            hist = [
                "/gps/hist/point 0.0 0.0010030000000000002",
                "/gps/hist/point 0.1282282715750936 0.003062",
                "/gps/hist/point 0.19234240736264038 0.00505",
                "/gps/hist/point 0.2564565431501872 0.007045",
                "/gps/hist/point 0.32057067893773394 0.009104",
                "/gps/hist/point 0.38468481472528077 0.010933",
                "/gps/hist/point 0.4487989505128276 0.012954",
                "/gps/hist/point 0.5129130863003744 0.014973",
                "/gps/hist/point 0.5770272220879211 0.016564",
                "/gps/hist/point 0.6411413578754679 0.018241000000000004",
                "/gps/hist/point 0.7052554936630148 0.019721",
                "/gps/hist/point 0.7693696294505615 0.021296",
                "/gps/hist/point 0.8334837652381083 0.022819",
                "/gps/hist/point 0.8975979010256552 0.024200000000000003",
                "/gps/hist/point 0.9617120368132019 0.025671",
                "/gps/hist/point 1.0258261726007487 0.027146999999999998",
                "/gps/hist/point 1.0899403083882955 0.02792",
                "/gps/hist/point 1.1540544441758422 0.029168",
                "/gps/hist/point 1.218168579963389 0.030045000000000002",
                "/gps/hist/point 1.2822827157509358 0.030493000000000003",
                "/gps/hist/point 1.3463968515384828 0.031058000000000002",
                "/gps/hist/point 1.4105109873260295 0.031564999999999996",
                "/gps/hist/point 1.4746251231135763 0.03187",
                "/gps/hist/point 1.538739258901123 0.032293",
                "/gps/hist/point 1.6028533946886698 0.031845",
                "/gps/hist/point 1.6669675304762166 0.032009",
                "/gps/hist/point 1.7310816662637634 0.031861",
                "/gps/hist/point 1.7951958020513104 0.031464",
                "/gps/hist/point 1.8593099378388571 0.030916999999999997",
                "/gps/hist/point 1.9234240736264039 0.030314",
                "/gps/hist/point 1.9875382094139507 0.029705",
                "/gps/hist/point 2.0516523452014974 0.028949999999999997",
                "/gps/hist/point 2.1157664809890444 0.027985",
                "/gps/hist/point 2.179880616776591 0.026889999999999997",
                "/gps/hist/point 2.243994752564138 0.025942",
                "/gps/hist/point 2.3081088883516845 0.024414999999999996",
                "/gps/hist/point 2.3722230241392315 0.023016999999999996",
                "/gps/hist/point 2.436337159926778 0.021287999999999998",
                "/gps/hist/point 2.500451295714325 0.020052",
                "/gps/hist/point 2.5645654315018716 0.018493",
                "/gps/hist/point 2.6286795672894185 0.016702",
                "/gps/hist/point 2.6927937030769655 0.014903",
                "/gps/hist/point 2.756907838864512 0.012795",
                "/gps/hist/point 2.821021974652059 0.010917",
                "/gps/hist/point 2.8851361104396056 0.00908",
                "/gps/hist/point 2.9492502462271526 0.007055",
                "/gps/hist/point 3.013364382014699 0.005001",
                "/gps/hist/point 3.077478517802246 0.0031040000000000004",
                "/gps/hist/point 3.141592653589793 0.0011009999999999998",
            ]

            for h in hist:
                f.write(f"{h} \n")
            # f.write("/gps/hist/point 1.0472 1 \n")

        else:
            f.write("/gps/ang/type cos \n")
            f.write("/gps/ang/mintheta 0 deg \n")
            f.write("/gps/ang/maxtheta 90 deg \n")

        f.write("/gps/ene/type %s \n" % ene_type)
        if ene_type == "Mono":
            f.write("/gps/ene/mono %2.f keV \n" % ene_min_keV)
        else:
            f.write("/gps/ene/min %.2f keV \n" % ene_min_keV)
            f.write("/gps/ene/max %.2f keV \n" % ene_max_keV)
        f.write(f"/analysis/setFileName {fname_tag} \n")

        f.write("/analysis/h1/set 1 100 100 1000 keV \n")
        f.write(
            f"/analysis/h2/set 1 100 {pos[0]-5} {pos[0]+5} cm none linear 100 {pos[1]-5} {pos[1]+5} cm none linear \n"
        )
        f.write(
            f"/analysis/h2/set 2 100 {pos[1]-5} {pos[1]+5} cm none linear 100 {pos[2]-5} {pos[2]+5} cm none linear \n"
        )
        f.write(
            f"/analysis/h2/set 3 100 {pos[2]-5} {pos[2]+5} cm none linear 100 {pos[0]-5} {pos[0]+5} cm none linear \n"
        )
        f.write(
            "/analysis/h2/set 4 120 0 360 deg none linear 100 -1 1 none none linear \n"
        )
        f.write(
            "/analysis/h2/set 5 120 0 360 deg none linear  90 0 180 deg none linear \n"
        )
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
