import numpy as np
from plot_settings import *

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

"""plot the size of the fully and partialy coded fov
"""

def fcfov_plane(detector_size_mm: float, mask_size_mm: float, mask_detector_distance_mm: float, mask_plane_distance_mm: float):
    detector_diagonal_mm = detector_size_mm * np.sqrt(2)
    mask_diagonal_mm = mask_size_mm * np.sqrt(2)

    detector_half_diagonal_mm = detector_diagonal_mm / 2
    mask_half_diagonal_mm  = mask_diagonal_mm / 2

    # FCFOV half angle
    theta_fcfov_deg = np.rad2deg(np.arctan((mask_half_diagonal_mm - detector_half_diagonal_mm ) / mask_detector_distance_mm))
    print(theta_fcfov_deg, "half angle")

    # pcfov
    fov = np.rad2deg(np.arctan((detector_diagonal_mm + (mask_half_diagonal_mm - detector_half_diagonal_mm))  / mask_detector_distance_mm))
    print("PCFOV", fov - theta_fcfov_deg, "Half angle pcfov")
    # project this to a distance
    plane_distance_to_detector_mm = mask_detector_distance_mm + mask_plane_distance_mm

    additional_diagonal_mm = np.tan(np.deg2rad(theta_fcfov_deg)) * plane_distance_to_detector_mm

    plane_diagonal_mm = (additional_diagonal_mm + detector_half_diagonal_mm) * 2

    plane_side_length_mm = plane_diagonal_mm / np.sqrt(2)

    # geant asks for half side length

    plane_half_side_length_mm = plane_side_length_mm / 2

    print(f"FCFOV square plane should be half side length = {plane_half_side_length_mm} mm at a distance {plane_distance_to_detector_mm} mm from detector")
    
    pcfov = fov-theta_fcfov_deg
    return pcfov, theta_fcfov_deg

detector_size_mm = 49.56
mask_size_mm = 98.28
mask_detector_distance_mm = 34.7
element_size_mm = 0.84
mask_plane_distance_mm = 0.15 + 1.0074 + 0.5

mask_detector_distance_mms = np.linspace(1,100,50)

fcfovs = []
pcfovs = []
fovs = []
for mask_detector_distance_mm in mask_detector_distance_mms:
    pcfov, fcfov = fcfov_plane(detector_size_mm, mask_size_mm, mask_detector_distance_mm, mask_plane_distance_mm)
    fcfovs.append(fcfov)
    pcfovs.append(pcfov)
    fovs.append(pcfov/(fcfov/2))

fig, ax = plt.subplots()
ax.set_facecolor('black')
fig.patch.set_facecolor('black')
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.spines['left'].set_color('white')

ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.yaxis.label.set_color('white')
ax.xaxis.label.set_color('white')


plt.plot(mask_detector_distance_mms,pcfovs,color='white')
plt.plot(mask_detector_distance_mms,fcfovs,color='blue')
plt.plot(mask_detector_distance_mms,fovs,color='pink')

plt.xlabel('F-number')
plt.ylabel('PCFOV/FCFOV')

plt.savefig('fov_ratio.png',dpi=300)