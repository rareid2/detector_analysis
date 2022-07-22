def write_det_config(det1_thickness_um, det_gap_mm, win_thickness_um, det_size_cm):
    f = open("../EPAD_geant4/src/det_config.txt", "w+")
    # thickness in um
    f.write(str(det1_thickness_um) + "\n")
    # gap in mm
    f.write(str(det_gap_mm) + "\n")
    # thickness of window in um
    f.write(str(win_thickness_um) + "\n")
    # size of the detector
    f.write(str(det_size_cm))

    f.close()
    print("Wrote det config file to src folder")

    return


def write_ca_config(
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
    f.write(aperture_path)

    f.close()
    print("Wrote ca config file to src folder")

    return
