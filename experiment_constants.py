import numpy as np
from datetime import date


# --------------------------------------- EXPERIMENT SETUP -----------------------------------------------
plastic_distance = 0.14 # cm
distance_uncertainty = 0.005 # cm
rho_air = 0.00114 # g/cm^3 dry air in Boulder CO
rho_pla = (1.24*0.71 + 1.18*0.29) # g/cm^3 weighted density of PLA(1mm) and plexiglas(0.4mm)

# ---------------------------------------- MINIPIX EDU ---------------------------------------------------
n_pixels = 256
detector_side_length_cm = 1.408 # cm
sigma_energy = 0.029118 # from energy calibration @ 60keV

# ---------------------------------------- Cobalt-57 ----------------------------------------------------
emission_peaks_co57 = np.array([6.266, 6.391, 6.404, 7.058, 7.058, 7.108, 7.112, 14.413])
branching_ratios_co57 = np.array([1.69e-5, 16.4, 32.6, 1.99, 3.88, 0.00206, 2.20e-7, 9.16])/100

bq_1_26_2023_co57 = 140628.6 # more active source
half_life_co57 = 271.79 #days
half_life_co57_unc = 0.009 # days  
d0_co57 = date(2023,1,26)

# ---------------------------------------- Barium-133 ---------------------------------------------------
emission_peaks_ba133 = np.array([30.27,30.625,30.973,34.920,34.987,35.252,35.818,35.907])
branching_ratios_ba133 = np.array([0.00401,34.9,64.5,5.99,11.6,0.123,3.58,0.74])/100

bq_1_26_2023_ba133 = 275734.6 
half_life_ba133 = 3838.6724 #days
half_life_ba133_unc = 1.8262 # days
d0_ba133 = date(2023,1,26)

# ---------------------------------------- Cesium-137 ---------------------------------------------------
emission_peaks_cs137 = np.array([31.452, 31.817, 32.194, 36.304, 36.378, 36.652, 37.255, 37.349])
branching_ratios_cs137 = np.array([0.000263, 2.04, 3.76, 0.352, 0.680, 0.0079, 0.215, 0.0481])/100

bq_1_30_2023_cs137 = 328522.4
half_life_cs137 = 10982.7668  # days
half_life_cs137_unc = 1.09572 # days
d0_cs137 = date(2023,1,30)

# ---------------------------------------- Europium-152 --------------------------------------------------
emission_peaks_eu152 = np.array([39.097, 39.522, 40.118])
branching_ratios_eu152 = np.array([0.00536, 21.1, 38.3])/100

bq_1_30_2023_eu152 = 28584.5
half_life_eu152 = 4944.473024  # days
half_life_eu152_unc = 0.219144  #days
d0_eu152 = date(2023,1,30)

# -------------------------------------- Cadmium-109 -----------------------------------------------------
emission_peaks_cd109 = np.array([21.708, 21.990, 22.163, 24.912, 24.943, 25.144, 25.455, 25.511])
branching_ratios_cd109 = np.array([0.00122, 29.5, 55.7, 4.76, 9.2, 0.067, 2.30, 0.487])/100

half_life_cd109 = 462.6 # days
half_life_cd109_unc = 0.004 # days