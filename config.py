def write_det_config(
    det1_thickness_um: float,
    det_gap_mm: float,
    win_thickness_um: float,
    det_size_cm: float,
) -> None:
    """
    write a config file that sets up detector geometry

    params:
        det1_thickness_um: thickness of front detector in um
        det_gap_mm:        gap between detectors in mm
        win_thickness_um:  thickness of window in um
    returns:
    """

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
    n_elements: int,
    mask_thickness_um: float,
    mask_gap_cm: float,
    element_size_mm: float,
    aperture_filename: str,
) -> None:
    """
    write a config file that sets up coded aperture geometry

    params:
        n_elements:        total number of elements in mask design
        mask_thickness_um: thickness of mask in um
        mask_gap_cm:       gap between mask and front detector in cm
        element_size_mm:   size of each aperture element in mm
        aperture_filename: filename containing MURA design structure
    returns:
    """

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
