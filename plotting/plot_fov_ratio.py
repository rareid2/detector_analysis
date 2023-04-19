import numpy as np
from plot_settings import *

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

    """plot the size of the fully and partialy coded fov
    """

det_size_cm = 1.408 # cm
distances = np.flip(np.linspace(0.1*2*det_size_cm,10*2*det_size_cm,50))
fnumber = distances / (2*det_size_cm)

hex_list = ["#023047","#219EBC","#FFB703","#FB8500","#F15025"]
triangle = 5 * det_size_cm / 2
triangle_2 = det_size_cm / 2
fov = np.arctan(triangle / distances)
fcfov = np.arctan(triangle_2 / distances)

pcfov = fov - fcfov
fig, ax = plt.subplots()
#plt.plot(fnumber,2*np.rad2deg(pcfov))
plt.plot(fnumber,2*np.rad2deg(fcfov),color=hex_list[-1])
plt.plot(fnumber,2*np.rad2deg(fov),color=hex_list[0])
ax.fill_between(fnumber, 2*np.rad2deg(fcfov), 0, color=hex_list[-1], alpha=.1)
ax.fill_between(fnumber, 2*np.rad2deg(pcfov)+2*np.rad2deg(fcfov), 2*np.rad2deg(fcfov), color=hex_list[1], alpha=.1)
plt.xlabel('F-number')
plt.ylabel('FOV [deg]')
plt.xlim([0,6])
plt.savefig('fov.png',dpi=300)

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


plt.plot(fnumber,2*np.rad2deg(pcfov)/(2*np.rad2deg(fcfov)),color='white')
plt.xlabel('F-number')
plt.ylabel('PCFOV/FCFOV')
plt.xlim([0,6])
plt.savefig('fov_ratio.png',dpi=300)