import matplotlib.pyplot as plt
import numpy as np
from plot_settings import *

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})


"""
plot the bounce loss cone and the gyroradii as a function of L-shell 
"""
L = np.linspace(1,7,50)
Re = 6378.14 # km
hm = 100 # km

ksi = (Re + hm) / (L*Re)

bounce_loss_cone = np.rad2deg(np.arcsin(np.sqrt(ksi ** 3 / (1 + 3*(1- ksi)))))

lshells_outer = []
lshells_inner = []

gryo_radiis_high = []
gryo_radiis_low = []
gryo_radiis_mid = []
gryo_radiis_mid_mid = []
gryo_radiis_mid_hi = []


gryo_radiis_low_inner_lo = []
gryo_radiis_low_inner_mid = []
gryo_radiis_low_inner_hi = []


for lshell in L:
    if lshell > 2.4:
        lshells_outer.append(lshell)
        
        gryo_radii = Re*np.sqrt(10) * (lshell/38.9)**3
        print(gryo_radii, 'high outer')
        gryo_radiis_high.append(gryo_radii)

        gryo_radii = Re*np.sqrt(0.1) * (lshell/38.9)**3
        print(gryo_radii, 'low outer')
        gryo_radiis_low.append(gryo_radii)

        gryo_radii = Re*np.sqrt(0.316) * (lshell/38.9)**3
        print(gryo_radii, 'low outer')
        gryo_radiis_mid.append(gryo_radii)

        gryo_radii = Re*np.sqrt(1) * (lshell/38.9)**3
        print(gryo_radii, 'low outer')
        gryo_radiis_mid_mid.append(gryo_radii)

        gryo_radii = Re*np.sqrt(3.16) * (lshell/38.9)**3
        print(gryo_radii, 'low outer')
        gryo_radiis_mid_hi.append(gryo_radii)

    else:       
        lshells_inner.append(lshell)

        gryo_radii = Re*np.sqrt(0.01) * (lshell/38.9)**3
        print(gryo_radii, 'low inner')
        gryo_radiis_low_inner_lo.append(gryo_radii)

        gryo_radii = Re*np.sqrt(0.0316) * (lshell/38.9)**3
        print(gryo_radii, 'low inner')
        gryo_radiis_low_inner_mid.append(gryo_radii)

        gryo_radii = Re*np.sqrt(0.1) * (lshell/38.9)**3
        print(gryo_radii, 'low inner')
        gryo_radiis_low_inner_hi.append(gryo_radii)

fig,ax = plt.subplots()
#plt.plot(L, bounce_loss_cone, color=hex_list[-1])
#plt.ylabel(r'equatorial $\alpha_{lc}^\degree$')
plt.xlabel('L-shell')

print(bounce_loss_cone[16],bounce_loss_cone[33])

#ax=ax.twinx()

ax.plot(lshells_inner, gryo_radiis_low_inner_lo, '--',color=hex_list[2])
ax.plot(lshells_inner, gryo_radiis_low_inner_mid, '--',color="#DF9D01")
ax.plot(lshells_inner, gryo_radiis_low_inner_hi, '--',color="#B78001")

ax.plot(lshells_outer, gryo_radiis_low, '--',color="#97DCED")
ax.plot(lshells_outer, gryo_radiis_mid, '--',color="#63CAE3")
ax.plot(lshells_outer, gryo_radiis_mid_mid, '--',color="#25AED0")
ax.plot(lshells_outer, gryo_radiis_mid_hi, '--',color="#18748B")


ax.plot(lshells_outer, gryo_radiis_high, '--',color="#0A122A")
ax.set_yscale('log')

ax.set_ylabel('equatorial gyroradii [km]')

plt.savefig('plotting/prospectus_loss_cone.png', dpi=500, bbox_inches='tight')