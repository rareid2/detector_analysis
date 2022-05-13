import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def align_vectors(a, b):
    #b = b / np.linalg.norm(b) # normalize a
    #a = a / np.linalg.norm(a) # normalize b
    v = np.cross(a, b)
    # s = np.linalg.norm(v)
    c = np.dot(a, b)

    v1, v2, v3 = v
    h = 1 / (1 + c)

    Vmat = np.array([[0, -v3, v2],
                  [v3, 0, -v1],
                  [-v2, v1, 0]])

    R = np.eye(3, dtype=np.float64) + Vmat + (Vmat.dot(Vmat) * h)
    return R

detector_loc = np.array([0,0,113])
src = np.array([10,10,-113])
norm_d = np.linalg.norm(detector_loc - src)
normal = (detector_loc - src) / norm_d

d = np.linalg.norm(src)

# create x,y
xx, yy = np.meshgrid(np.arange(-100,100), np.arange(-100,100))

# calculate corresponding z
z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]

# plot the surface
plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(xx, yy, z, alpha=0.2)

# take the cross product???
z_ax = np.array([0,0,1])
cc = np.linalg.norm(np.cross(z_ax,normal))
dd = np.dot(z_ax,normal)
r = np.array([[dd,-cc,0],[cc,dd,0],[0,0,1]])

u = z_ax
v = (normal - (dd * z_ax)) / np.linalg.norm(normal - (dd * z_ax))
w = np.cross(normal,z_ax)
F = np.array([u,v,w])
F_inv = np.linalg.inv(F)
new_vec = F * z_ax * F_inv


new_vec = -1*new_vec[0,:]
R = align_vectors(z_ax, normal)
print(R)

new_vec = R * z_ax

plt3d.scatter(src[0],src[1],src[2])
plt3d.scatter(detector_loc[0],detector_loc[1],detector_loc[2])
plt3d.quiver(src[0],src[1],src[2],500*normal[0],500*normal[1],500*normal[2])
plt3d.quiver(src[0],src[1],src[2],50*new_vec[0],50*new_vec[1],50*new_vec[2])
#plt3d.quiver(detector_loc[0],detector_loc[1],detector_loc[2],0,0,100)
plt.show()
plt.show()
plt.close()