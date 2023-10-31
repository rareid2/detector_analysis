import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_sr(focal_length, x_extent, y_extent, px_loc, sphere_radius
):
    # input the location of the pixel in x and y and the extent in x and y
    # sphere needs to be centered at the pinhole

    pixel0 = np.array([px_loc[0] - (x_extent / 2), px_loc[1] - (y_extent / 2),  -1 * focal_length])
    pixel1 = np.array([px_loc[0] + (x_extent / 2), px_loc[1] - (y_extent / 2),  -1 * focal_length])
    pixel2 = np.array([px_loc[0] + (x_extent / 2), px_loc[1] + (y_extent / 2),  -1 * focal_length])
    pixel3 = np.array([px_loc[0] - (x_extent / 2), px_loc[1] + (y_extent / 2),  -1 * focal_length])

    # location of the pinhole
    pinhole = np.array([0, 0, 0])

    # Calculate the vector from point1 to point2
    vector0 = pinhole - pixel0
    vector1 = pinhole - pixel1
    vector2 = pinhole - pixel2
    vector3 = pinhole - pixel3

    # Define the ray's origin and direction
    pixels = [pixel0, pixel1, pixel2, pixel3]
    vectors = [vector0, vector1, vector2, vector3]

    def ray_sphere_intersection(
        ray_origin, ray_direction, sphere_center, sphere_radius
    ):
        # Calculate the vector from the ray's origin to the sphere's center
        oc = ray_origin - sphere_center

        # Calculate coefficients for the quadratic equation
        a = np.dot(ray_direction, ray_direction)
        b = 2.0 * np.dot(ray_direction, oc)
        c = np.dot(oc, oc) - sphere_radius**2

        # Calculate the discriminant
        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            return None  # No intersection

        # Calculate the two solutions for t (parameter along the ray)
        t1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2.0 * a)

        # Choose the smaller positive t value as the intersection point
        t = min(t1, t2)

        # Calculate the intersection point
        intersection_point = ray_origin + t * ray_direction

        return intersection_point
    

    intersection_points = []        
    theta_extents = []
    phi_extents = []
    points = []

    for pixel, vector in zip(pixels,vectors):
        # ray origin is from pixel location
        ray_origin = pixel
        ray_direction = vector / np.linalg.norm(vector)

        # Define the sphere's center and radius
        sphere_center = np.array([0.0, 0.0, 0.0])

        # Calculate the intersection point on the surface of the sphere
        intersection_point = ray_sphere_intersection(
            ray_origin, ray_direction, sphere_center, sphere_radius
        )

        # find the theta and phi extent from the intersection points

        x = intersection_point[0]
        y = intersection_point[1]
        z = intersection_point[2]
        points.append((x,y,z))

        r = math.sqrt(x**2 + y**2 + z**2)
        phi = math.atan2(y, x)
        theta = math.acos(z / r)
        
        # change for expected coordinates
        theta = np.pi - theta
        theta_extents.append(theta)
        if phi < 0:
            phi = 2*np.pi + phi
            phi_extents.append(phi)
        elif phi == 0:
            pass
        else:
            phi_extents.append(phi)


    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 0 + r * np.outer(np.cos(u), np.sin(v))
    y = 0 + r * np.outer(np.sin(u), np.sin(v))
    z = 0 + r * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='b', alpha=0.5)
    for point in points:
        ax.scatter(*point, color='r', s=50, label='Point')
    plt.show()
    """

    # unpack into extents
    theta_min = min(np.array(theta_extents))
    theta_max = max(np.array(theta_extents))

    phi_min = min(np.array(phi_extents))
    phi_max = max(np.array(phi_extents))
    
    # consider wrap around, find angular distance of pixel
    if phi_max - phi_min > np.pi:
        phi_extent = ((np.pi*2) - phi_max) + phi_min
    else:
        phi_extent = phi_max - phi_min
    
    sr = phi_extent * (np.cos(theta_min) - np.cos(theta_max))
    return sr, phi_extent