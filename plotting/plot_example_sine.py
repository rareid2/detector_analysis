import matplotlib.pyplot as plt
import numpy as np
from plot_settings import *

plt.rcParams.update({'font.size': 14})

# generate 2 2d grids for the x & y bounds
y, x = np.meshgrid(np.linspace(0, np.pi, 100), np.linspace(0, np.pi, 100))

z = np.sin(x)
# x and y are bounds, so z should be the value *inside* those bounds.
# Therefore, remove the last value from the z array.
z = z[:-1, :-1]
z_min, z_max = -np.abs(z).max(), np.abs(z).max()

fig, ax = plt.subplots()

c = ax.pcolormesh(x, y, z, cmap=cmap, vmin=0, vmax=1)

# set the limits of the plot to the limits of the data
ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(c, ax=ax)

plt.savefig('sin_heatmap.png',dpi=300)

fig, ax = plt.subplots()
z_sum = np.sum(z,axis=1)
x = np.linspace(0, np.pi, 100)
x = np.rad2deg(x[:-1])
y = z_sum

xs = [x[0] - 0.5]
ys = [0]
for i in range(len(x)):
    xs.append(x[i] - 0.5)
    xs.append(x[i] + 0.5)
    ys.append(y[i])
    ys.append(y[i])
xs.append(x[-1] + 0.5)
ys.append(0)
plt.plot(xs, ys, color="#023047")

# optionally color the area below the curve
plt.fill_between(xs, 0, ys, color="#d6f6eb")


#plt.scatter(np.rad2deg(xx[:-1]),z_sum)
plt.xlabel('deg')
plt.ylabel('total integrated flux')
#ax.bar(np.rad2deg(xx[:-1]),z_sum,width=1.8,align='center',alpha=0)
plt.savefig('z_sum.png',dpi=300)