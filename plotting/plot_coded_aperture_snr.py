import numpy as np
import matplotlib.pyplot as plt
from plot_settings import *

nt = 31**2
rho = 0.5

psi = np.linspace(1/nt,1,500)
ksi = np.logspace(-1,1,500)

x, y = np.meshgrid(psi,ksi)

snr = np.sqrt(nt)*np.sqrt(rho*(1-rho))*np.sqrt(x+y) / np.sqrt(rho + ((1 - 2*rho)*x) + y)
#snr = np.sqrt(17*x+y) / np.sqrt(1+8*y)

fig = plt.figure()
levels = np.linspace(6,21,11)
plt.contourf(y,x,snr, cmap = cmap,levels=levels,extend='both')
plt.xscale('log')
cbar = plt.colorbar()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.xlabel(r'$\psi$',fontsize=16)
import matplotlib.ticker as tick
cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
cbar.ax.tick_params(labelsize=14)

plt.savefig('plotting/prospectus_snr.png',dpi=500)

