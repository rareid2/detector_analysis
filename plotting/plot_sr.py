import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys

sys.path.insert(1, "../detector_analysis")
from plotting.calculate_solid_angle import get_sr
# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# general detector design
det_size_cm = 2  # cm
pixel = 2.5 * 0.1  # cm
mask_size_cm = 3

# Create a sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
r = 2.5
x = r * np.outer(np.cos(u), np.sin(v))
y = r * np.outer(np.sin(u), np.sin(v))
z = r * np.outer(np.ones(np.size(u)), np.cos(v))
get_sr(2, .25/2, .25/2, (-0.6,-0.6), 2.5)

# Plot the sphere
ax.plot_surface(x, y, z, color='b', alpha=0.2)

# Define the vertices of a plane
imaginary_plane = -2.
plane_vertices = np.array([
    [-det_size_cm / 2, -det_size_cm / 2, imaginary_plane],
    [-det_size_cm / 2, det_size_cm / 2, imaginary_plane],
    [det_size_cm / 2, det_size_cm / 2, imaginary_plane],
    [det_size_cm / 2, -det_size_cm / 2, imaginary_plane],
])


pinhole_vertices = np.array([
    [-mask_size_cm / 2, -mask_size_cm / 2, 0],
    [-mask_size_cm / 2, mask_size_cm / 2, 0],
    [mask_size_cm / 2, mask_size_cm / 2, 0],
    [mask_size_cm / 2, -mask_size_cm / 2, 0],
])

holes_size = 0.075
hole_vertices = np.array([
    [-1*holes_size, -1*holes_size, 0],
    [-1*holes_size, holes_size, 0],
    [holes_size, holes_size, 0],
    [holes_size, -1*holes_size, 0],
])
px_offset = -0.6
px_vertices = np.array([
    [-pixel / 2 + px_offset, -pixel / 2 + px_offset, imaginary_plane],
    [-pixel / 2 + px_offset, pixel / 2+ px_offset, imaginary_plane],
    [pixel / 2 + px_offset, pixel / 2+ px_offset, imaginary_plane],
    [pixel / 2 + px_offset, -pixel / 2+ px_offset, imaginary_plane],
])
# Define the origin and the vector coordinates
for pxv in px_vertices:
    origin = pxv
    vector = r*np.array([0-pxv[0], 0-pxv[1], 0-imaginary_plane])  # Replace with your vector coordinates
    ax.quiver(*origin, *vector, color='b', label='Vector', linewidth=0.5)



# Connect the vertices to create the plane
plane_faces = [[plane_vertices[0], plane_vertices[1], plane_vertices[2], plane_vertices[3]]]
pinhole_faces = [[pinhole_vertices[0], pinhole_vertices[1], pinhole_vertices[2], pinhole_vertices[3]]]
hole_faces = [[hole_vertices[0], hole_vertices[1], hole_vertices[2], hole_vertices[3]]]
px_faces = [[px_vertices[0], px_vertices[1], px_vertices[2], px_vertices[3]]]

# Plot the plane
ax.add_collection3d(Poly3DCollection(plane_faces, facecolors='grey', linewidths=1, zorder=0, alpha=0.5))
ax.add_collection3d(Poly3DCollection(pinhole_faces, facecolors='grey', linewidths=1, zorder=1, alpha=0.5))
ax.add_collection3d(Poly3DCollection(hole_faces, facecolors='black', linewidths=1, zorder=5))
ax.add_collection3d(Poly3DCollection(px_faces, facecolors='black', linewidths=1, zorder=8))


# Create a small patch
v = np.linspace(2 * np.pi - 0.36320808410341954, 2 * np.pi - 0.4380973339657257, 100)
u = np.linspace(3.8231984765301985, 4.030783157444285, 100)

x = r * np.outer(np.cos(u), np.sin(v))
y = r * np.outer(np.sin(u), np.sin(v))
z = r * np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the sphere
ax.plot_surface(x, y, z, color='r', alpha=1)

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set axis limits for a better view
ax.set_xlim([-(r+0), (r+0)])
ax.set_ylim([-(r+0), (r+0)])
ax.set_zlim([-(r+0), (r+0)])

# Display the plot
plt.show()