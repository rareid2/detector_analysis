import numpy as np
import matplotlib.pyplot as plt


def inverse_stereographic_projection(projected_points):
    # Calculate the squared norm of the projected points
    squared_norms = np.sum(projected_points**2, axis=1)

    # Calculate the 3D coordinates using the inverse projection formula
    den = squared_norms + 1
    points_3d = np.column_stack(
        [
            (2 * projected_points[:, 0]) / den,
            (2 * projected_points[:, 1]) / den,
            (-squared_norms + 1) / den,
        ]
    )

    return points_3d


# Example projected 2D points on the stereographic plane
projected_points = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

# Perform the inverse stereographic projection
points_3d = inverse_stereographic_projection(projected_points)

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi / 2, 50)
u, v = np.meshgrid(u, v)
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)

# Plot the 3D points on the unit sphere
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])
ax.plot_surface(x, y, z, color="b", alpha=0.5)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_aspect("equal", "box")
ax.set_title("Inverse Stereographic Projection")
plt.savefig("test.png")
