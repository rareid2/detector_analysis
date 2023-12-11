import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sphericalpolygon import Sphericalpolygon


def ray_sphere_intersection(
    ray_origin,
    ray_direction,
    sphere_center,
    sphere_radius,
    calc_pinhole=False,
):
    # Calculate the vector from the ray's origin to the sphere's center
    oc = ray_origin - sphere_center

    # Calculate coefficients for the quadratic equation
    if calc_pinhole:
        a = np.dot(ray_direction, ray_direction)
        b = 2.0 * np.dot(ray_direction, oc)
        c = np.dot(oc, oc) - sphere_radius**2
    else:
        a = np.sum(ray_direction * ray_direction, axis=1)
        b = 2.0 * np.sum(ray_direction * oc, axis=1)
        c = np.sum(oc * oc, axis=1) - sphere_radius**2

    # Calculate the discriminant
    discriminant = b**2 - 4 * a * c

    # if discriminant < 0:
    #    return None  # No intersection

    # Calculate the two solutions for t (parameter along the ray)
    t1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
    t2 = (-b + np.sqrt(discriminant)) / (2.0 * a)

    # Choose the smaller positive t value as the intersection point
    if calc_pinhole:
        t = min(np.abs(t1), np.abs(t2))
    else:
        t = np.minimum(np.absolute(t1), np.absolute(t2))

    # Calculate the intersection point
    if calc_pinhole:
        intersection_point = ray_origin + ray_direction * t
    else:
        intersection_point = ray_origin + ray_direction * t[:, np.newaxis]

    return intersection_point


def get_sr(
    focal_length,
    x_extent,
    y_extent,
    px_loc,
    sphere_radius,
    mask,
    calc_pinhole=False,
    girard=False,
):
    # input the location of the pixel in x and y and the extent in x and y
    # sphere needs to be centered at the pinhole

    # input pixel corners
    pixel0 = np.array(
        [px_loc[0] + (x_extent / 2), px_loc[1] - (y_extent / 2), -1 * focal_length]
    )
    pixel1 = np.array(
        [px_loc[0] - (x_extent / 2), px_loc[1] - (y_extent / 2), -1 * focal_length]
    )
    pixel2 = np.array(
        [px_loc[0] - (x_extent / 2), px_loc[1] + (y_extent / 2), -1 * focal_length]
    )
    pixel3 = np.array(
        [px_loc[0] + (x_extent / 2), px_loc[1] + (y_extent / 2), -1 * focal_length]
    )
    pixels = np.array([pixel0, pixel1, pixel2, pixel3])
    # repeat by 4

    # get vectors from each mask element 4x (mask element / 2)
    if calc_pinhole == False:
        all_vectors = reverse_raytrace(px_loc, x_extent, y_extent, focal_length, mask)
        pixels = np.repeat(pixels, (len(all_vectors) // 4), axis=0)
    else:
        pinhole = np.array([0.0, 0.0, 0.0])
        vector0 = pinhole - pixel0
        vector1 = pinhole - pixel1
        vector2 = pinhole - pixel2
        vector3 = pinhole - pixel3
        vectors = [vector0, vector1, vector2, vector3]

    # Define the sphere's center and radius

    if calc_pinhole:
        intersection_points = []
        sphere_center = np.array([0.0, 0.0, 0.0])
        for pixel, vector in zip(pixels, vectors):
            ray_origin = pixel
            ray_direction = vector / np.linalg.norm(vector)
            intersection_point = ray_sphere_intersection(
                ray_origin,
                ray_direction,
                sphere_center,
                sphere_radius,
                calc_pinhole=True,
            )
            intersection_points.append(intersection_point)
    else:
        ray_origin = pixels
        sphere_center = np.tile(np.array([0.0, 0.0, 0.0]), (len(all_vectors), 1))
        ray_direction = all_vectors / np.linalg.norm(all_vectors, axis=1)[:, None]

        # Calculate the intersection point on the surface of the sphere
        intersection_point = ray_sphere_intersection(
            ray_origin, ray_direction, sphere_center, sphere_radius, calc_pinhole=False
        )

    total_solid_angle = 0
    # now we need to index each one corner at a time
    if calc_pinhole == False:
        for i in range((len(all_vectors) // 4)):
            itps = intersection_point[i :: (len(all_vectors) // 4)]
            theta_extents = []
            phi_extents = []
            units_vtxs = []
            vtx_norms = []
            interior_angles = []
            alpha = 0
            for itp in itps:
                x = itp[0]
                y = itp[1]
                z = itp[2]

                if girard:
                    unit_pt = itp / np.linalg.norm(itp)
                    units_vtxs.append(unit_pt)
                else:
                    # normal differential calc
                    r = math.sqrt(x**2 + y**2 + z**2)
                    phi = math.atan2(y, x)
                    theta = math.acos(z / r)

                    # change for expected coordinates
                    theta = np.pi / 2 - theta
                    theta_extents.append(np.rad2deg(theta))

                    phi = (np.rad2deg(phi) + 360) % 360
                    phi_extents.append(phi)
            if girard:
                pts = [0, 1, 2, 3, 0]
                for i in pts[:4]:  # go through first 4
                    vtx_norm = np.cross(units_vtxs[i], units_vtxs[pts[i + 1]])
                    # cross product of two unit vectors is not a unit vector
                    vtx_norms.append(vtx_norm)
                for i in pts[1:]:
                    n1 = np.linalg.norm(vtx_norms[i])
                    n2 = np.linalg.norm(vtx_norms[pts[i - 1]])
                    f = 1
                    alpha += f * np.arccos(
                        -1 * np.dot(vtx_norms[pts[i - 1]], vtx_norms[i]) / (n1 * n2)
                    )
                    interior_angles.append(alpha)
                sr = alpha - (2 * np.pi)
                total_solid_angle = sr

            else:
                # unpack into extents
                theta_min = min(np.array(theta_extents))
                theta_max = max(np.array(theta_extents))

                phi_min = min(np.array(phi_extents))
                phi_max = max(np.array(phi_extents))

                # consider wrap around, find angular distance of pixel
                if phi_max - phi_min > 180:
                    # when wrapped over 0, the largest value is actually second largest and the smallest is second smallest
                    phi_max = np.partition(phi_extents, -2)[-2]
                    phi_min = np.partition(phi_extents, -2)[1]
                    phi_extent = ((360) - phi_max) + phi_min
                else:
                    phi_extent = phi_max - phi_min

                sr = np.deg2rad(phi_extent) * (
                    np.cos(np.deg2rad(theta_min)) - np.cos(np.deg2rad(theta_max))
                )
                total_solid_angle += sr
    else:  # doing pinhole method
        theta_extents = []
        phi_extents = []
        units_vtxs = []
        vtx_norms = []
        interior_angles = []
        alpha = 0
        for itp in intersection_points:
            x = itp[0]
            y = itp[1]
            z = itp[2]
            if girard:
                unit_pt = itp / np.linalg.norm(itp)
                units_vtxs.append(unit_pt)
            else:
                r = math.sqrt(x**2 + y**2 + z**2)
                phi = math.atan2(y, x)
                theta = math.acos(z / r)

                # change for expected coordinates
                theta = np.pi / 2 - theta
                theta_extents.append(np.rad2deg(theta))

                phi = (np.rad2deg(phi) + 360) % 360
                phi_extents.append(phi)
        if girard:
            pts = [0, 1, 2, 3, 0]
            for i in pts[:4]:  # go through first 4
                vtx_norm = np.cross(units_vtxs[i], units_vtxs[pts[i + 1]])
                # cross product of two unit vectors is not a unit vector
                vtx_norms.append(vtx_norm)
            for i in pts[1:-1]:
                n1 = np.linalg.norm(vtx_norms[i])
                n2 = np.linalg.norm(vtx_norms[pts[i - 1]])
                f = 1
                alpha += f * np.arccos(
                    -1 * np.dot(vtx_norms[pts[i - 1]], vtx_norms[i]) / (n1 * n2)
                )
                interior_angles.append(
                    f
                    * np.arccos(
                        -1 * np.dot(vtx_norms[pts[i - 1]], vtx_norms[i]) / (n1 * n2)
                    )
                )
            alpha += interior_angles[1]
            sr = alpha - (2 * np.pi)
            total_solid_angle = sr
            print(sr)
        else:
            # unpack into extents
            theta_min = min(np.array(theta_extents))
            theta_max = max(np.array(theta_extents))

            phi_min = min(np.array(phi_extents))
            phi_max = max(np.array(phi_extents))

            # consider wrap around, find angular distance of pixel
            if phi_max - phi_min > 180:
                # when wrapped over 0, the largest value is actually second largest and the smallest is second smallest
                phi_max = np.partition(phi_extents, -2)[-2]
                phi_min = np.partition(phi_extents, -2)[1]
                phi_extent = ((360) - phi_max) + phi_min
            else:
                phi_extent = phi_max - phi_min

            sr = np.deg2rad(phi_extent) * (
                np.cos(np.deg2rad(theta_min)) - np.cos(np.deg2rad(theta_max))
            )
            total_solid_angle += sr

    return total_solid_angle


def reverse_raytrace(px_loc, x_extent, y_extent, focal_length, mask):
    # input pixel location and return rays from that pixel through each mask hole
    # mask should be a grid of 1 and 0s

    # TODO make dynamic
    cell_size_cm = 0.05  # cm

    # input pixel corners
    pixel0 = np.array(
        [px_loc[0] - (x_extent / 2), px_loc[1] - (y_extent / 2), -1 * focal_length]
    )
    pixel1 = np.array(
        [px_loc[0] + (x_extent / 2), px_loc[1] - (y_extent / 2), -1 * focal_length]
    )
    pixel2 = np.array(
        [px_loc[0] + (x_extent / 2), px_loc[1] + (y_extent / 2), -1 * focal_length]
    )
    pixel3 = np.array(
        [px_loc[0] - (x_extent / 2), px_loc[1] + (y_extent / 2), -1 * focal_length]
    )
    pixels = [pixel0, pixel1, pixel2, pixel3]

    # set pixel id so that mask is centered at 0,0,0
    mask_center = 60  # for mosiac with 0 through 121
    mask_coordinates = []
    for row_idx, row in enumerate(mask):
        for col_idx, value in enumerate(row):
            if value == 0:
                x = (col_idx - mask_center) * cell_size_cm
                y = (row_idx - mask_center) * cell_size_cm
                mask_coordinates.append((x, y))

    # now we have location of each mask element
    # need ot raytrace each corner through center of each mask element

    all_vectors = np.zeros((4 * len(mask_coordinates), 3))

    for ii, pixel in enumerate(pixels):
        # loop through each pixel and create coordinates
        all_vectors[
            (ii) * len(mask_coordinates) : (ii + 1) * len(mask_coordinates), 0
        ] = [mc[0] - pixel[0] for mc in mask_coordinates]
        all_vectors[
            (ii) * len(mask_coordinates) : (ii + 1) * len(mask_coordinates), 1
        ] = [mc[1] - pixel[1] for mc in mask_coordinates]
        all_vectors[
            (ii) * len(mask_coordinates) : (ii + 1) * len(mask_coordinates), 2
        ] = [0 - pixel[2] for mc in mask_coordinates]

    # all vectors stacked each pixel at a time
    return all_vectors


# quick example to get total FOV solid angle coverage
def get_total_sr():
    focal_length = 2.03333333
    pixel0 = np.array([(0.025 * 122 / 2), -1 * (0.025 * 122 / 2), -1 * focal_length])
    pixel1 = np.array(
        [-1 * (0.025 * 122 / 2), -1 * (0.025 * 122 / 2), -1 * focal_length]
    )
    pixel2 = np.array([-1 * (0.025 * 122 / 2), (0.025 * 122 / 2), -1 * focal_length])
    pixel3 = np.array([(0.025 * 122 / 2), (0.025 * 122 / 2), -1 * focal_length])
    pixels = np.array([pixel0, pixel1, pixel2, pixel3])

    pinhole = np.array([0.0, 0.0, 0.0])
    vector0 = pinhole - pixel0
    vector1 = pinhole - pixel1
    vector2 = pinhole - pixel2
    vector3 = pinhole - pixel3
    vectors = [vector0, vector1, vector2, vector3]

    intersection_points = []
    sphere_center = np.array([0.0, 0.0, 0.0])
    for pixel, vector in zip(pixels, vectors):
        ray_origin = pixel
        ray_direction = vector / np.linalg.norm(vector)
        intersection_point = ray_sphere_intersection(
            ray_origin,
            ray_direction,
            sphere_center,
            8,
            calc_pinhole=True,
        )
        intersection_points.append(intersection_point)

    units_vtxs = []
    for itp in intersection_points:
        x = itp[0]
        y = itp[1]
        z = itp[2]
        # take the norm of the points to get location on unit sphere
        unit_pt = itp / np.linalg.norm(itp)

        units_vtxs.append(unit_pt)

    pts = [0, 1, 2, 3, 0]
    vtx_norms = []
    interior_angles = []
    for i in pts[:4]:  # go through first 4
        vtx_norm = np.cross(units_vtxs[i], units_vtxs[pts[i + 1]])

        # cross product of two unit vectors is not a unit vector
        vtx_norms.append(vtx_norm)

    alpha = 0
    for i in pts[1:-1]:
        n1 = np.linalg.norm(vtx_norms[i])
        n2 = np.linalg.norm(vtx_norms[pts[i - 1]])

        f = 1

        alpha += f * np.arccos(
            -1 * np.dot(vtx_norms[pts[i - 1]], vtx_norms[i]) / (n1 * n2)
        )
        interior_angles.append(
            f * np.arccos(-1 * np.dot(vtx_norms[pts[i - 1]], vtx_norms[i]) / (n1 * n2))
        )

    # add in last one
    alpha += interior_angles[0]

    # got it! this should be the area on a unit sphere (sr)
    sr = alpha - (2 * np.pi)

    return sr


if __name__ == "__main__":
    get_total_sr()
