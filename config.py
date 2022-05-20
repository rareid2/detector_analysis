# config file to set up all the detector / coded aperture settings

# ------- constants to have ---------

# imported from GEANT
env_sizeXY = 2600  # cm
env_sizeZ = 1111  # cm
world_sizeXY = 1.1 * env_sizeXY
world_sizeZ = 1.1 * env_sizeZ
world_offset = env_sizeZ * 0.45
detector_placement = world_offset  # im cm

det1_thickness_um = 140  # um
det_gap_mm = 30  # mm
win_thickness_um = 100  # um

n_elements = 133
mask_thickness_um = 400  # um
mask_gap_cm = 3  # cm
element_size_mm = 0.66  # mm


def create_det_config(det1_thickness_um, det_gap_mm, win_thickness_um):
    f = open("../EPAD_geant4/src/det_config.txt", "w+")
    # thickness in um
    f.write(str(det1_thickness_um) + "\n")
    # gap in mm
    f.write(str(det_gap_mm) + "\n")
    # thickness of window in um
    f.write(str(win_thickness_um) + "\n")

    f.close()
    print("Wrote det config file to src folder")

    return


def create_ca_config(
    n_elements, mask_thickness_um, mask_gap_cm, element_size_mm, aperture_filename
):
    f = open("../EPAD_geant4/src/ca_config.txt", "w+")
    # n elements (total number for mosaic)
    f.write(str(n_elements) + "\n")
    # thickness of mask in um
    f.write(str(mask_thickness_um) + "\n")
    # gap between front of mask and detector in cm
    f.write(str(mask_gap_cm) + "\n")
    # size of elements in mm
    f.write(str(element_size_mm) + "\n")
    # aperture filename
    aperture_path = "./src/mask_designs/" + aperture_filename
    f.write(aperture_path + "\n")

    f.close()
    print("Wrote ca config file to src folder")

    return
