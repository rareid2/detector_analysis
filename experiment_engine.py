import os

# experiment data abs path
experiment_data_path = "/home/rileyannereid/workspace/geant4/experiment_data"


class ExperimentEngine:
    def __init__(
        self,
        isotope: str = "Cd109",
        frames: int = 100,
        exposure_s: float = 1.0,
        source_detector_cm: float = 44.66,
        mask_detector_cm: float = 5.0,
        data_folder: str = "2cm/",
        n_files: int = 25,
    ) -> None:

        self.isotope = isotope
        self.frames = frames
        self.exposure_s = exposure_s
        self.source_detector_cm = source_detector_cm
        self.mask_detector_cm = mask_detector_cm
        self.data_folder = os.path.join(experiment_data_path, data_folder)
        self.n_files = n_files
