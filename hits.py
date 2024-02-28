import os, math
import pandas as pd
import numpy as np
from experiment_engine import ExperimentEngine
from uncertainty import add_uncertainty
from shapely.geometry import Point, Polygon


class Hits:
    def __init__(
        self,
        fname: str = None,
        experiment: bool = False,
        experiment_geant4: bool = False,
        experiment_engine: ExperimentEngine = None,
        file_count: int = 0,
        txt_file: bool = False,
        nlines: int = None,
        nstart: int = None,
    ) -> None:
        # filename containing hits data
        self.fname = fname
        self.txt_hits = None
        # ---------------------- parse the hits file -----------------------

        # if physical experiment data its already digitized
        # base_filename = os.path.basename(self.fname)
        if experiment:
            # create file name from the distances and # of frames etc.
            # m = re.match(r'(.*)-(.*)frames-(.*)s-(.*)cm-sd-(.*)cm-md', base_filename)
            # experiment_setup = {
            #    "isotope": m.group(1),
            #    "frames": float(m.group(2)),
            #    "exposure_s": float(m.group(3).replace("pt", ".")),
            #    "source_detector_cm": float(m.group(4).replace("pt", ".")),
            #    "mask_detector_cm": float(m.group(5).replace("pt", ".")),
            # }
            fname = "%s-%04dframes-%dpt%ds-%02dpt%02dcm-sd-%dpt%02dcm-md-%d.txt" % (
                experiment_engine.isotope,
                experiment_engine.frames,
                experiment_engine.exposure_s,
                round(10 * (experiment_engine.exposure_s % 1)),
                experiment_engine.source_detector_cm,
                round(100 * (experiment_engine.source_detector_cm % 1)),
                experiment_engine.mask_detector_cm,
                round(100 * (experiment_engine.mask_detector_cm % 1)),
                file_count,
            )
            fname = f"cd109-test_{file_count:03d}.txt"
            fname_full_path = os.path.join(experiment_engine.data_folder, fname)
            f = open(fname_full_path, "r")

            # get the array of data in
            lines = [line.split() for line in f]
            lines_array = np.array(
                [
                    [int(line_element.strip("/n")) for line_element in line]
                    for line in lines
                ]
            ).astype(float)
            lines_sum = np.sum(
                np.array([np.sum(line_array) for line_array in lines_array])
            )
            f.close()

            # add to dict
            self.detector_hits = lines_array
            # self.hits_dict = {"data": lines_array}
            self.n_entries = len(np.where(lines_array > 0))
            # add another dictionary with experiment data (distances etc.)
            # self.experiment_setup = experiment_setup

            self.fname = fname

        # geant experiment is set up in a new coordinate system to import CAD files correctly
        elif experiment_geant4:
            self.detector_hits = pd.read_csv(
                self.fname,
                names=["x", "y", "z", "energy"],
                dtype={
                    "x": np.float64,
                    "y": np.float64,
                    "z": np.float64,
                    "energy": np.float64,
                },
                delimiter=",",
                on_bad_lines="skip",
                engine="c",
            )
            self.n_entries = len(self.detector_hits["energy"])

        elif txt_file:
            self.txt_hits = np.loadtxt(self.fname)
            self.n_entries = np.sum(self.txt_hits)

        # general geant4 simulations
        else:
            if nlines:
                self.detector_hits = pd.read_csv(
                    self.fname,
                    names=["det", "x", "y", "z", "energy", "ID", "name", "x0", "y0", "z0"],
                    dtype={
                        "det": np.int8,
                        "x": np.float64,
                        "y": np.float64,
                        "z": np.float64,
                        "energy": np.float64,
                        "ID": np.int8,
                        "name": str,
                        "x0": np.float64,
                        "y0": np.float64,
                        "z0": np.float64,
                    },
                    delimiter=",",
                    on_bad_lines="skip",
                    engine="c",
                    nrows=nlines,
                    skiprows=range(1,nstart+1))
            else:
                self.detector_hits = pd.read_csv(
                    self.fname,
                    names=["det", "x", "y", "z", "energy", "ID", "name", "x0", "y0", "z0"],
                    dtype={
                        "det": np.int8,
                        "x": np.float64,
                        "y": np.float64,
                        "z": np.float64,
                        "energy": np.float64,
                        "ID": np.int8,
                        "name": str,
                        "x0": np.float64,
                        "y0": np.float64,
                        "z0": np.float64,
                    },
                    delimiter=",",
                    on_bad_lines="skip",
                    engine="c",
                )
            self.n_entries = len(self.detector_hits["det"])

        if self.n_entries == 0:
            raise ValueError("No particles hits on any detector!")

        self.hits_dict = {}

        # get detector size if needed for uncertainty
        # self.det_size_cm = simulation_engine.det_size_cm

        return

    def get_both_det_hits(self) -> dict:
        """
        return a dictionary containing hits on both detectors

        params:
        returns:
            hits_dict: contains positions of hits for each detector in particle order, also has energies
        """

        array_counter = 0

        posX1 = []
        posY1 = []
        energies1 = []

        posX2 = []
        posY2 = []
        energies2 = []

        for count, el in enumerate(self.detector_hits["det"]):
            # pandas series can throw a KeyError if character starts line
            # TODO: replace this with parse command that doesn't import keyerror throwing lines
            while True:
                try:
                    pos1 = self.detector_hits["det"][count]
                    pos2 = self.detector_hits["det"][count + 1]

                    self.detector_hits["x"][count]
                    self.detector_hits["y"][count]

                    self.detector_hits["x"][count + 1]
                    self.detector_hits["y"][count + 1]

                except:
                    count = count + 1
                    if count == self.n_entries:
                        break
                    continue
                break

            # Checks if first hit detector == 1 and second hit detector == 2
            if np.equal(pos1, 1) & np.equal(pos2, 2):
                posX1.append(self.detector_hits["x"][count])
                posY1.append(self.detector_hits["y"][count])

                posX2.append(self.detector_hits["x"][count + 1])
                posY2.append(self.detector_hits["y"][count + 1])

                energies1.append(self.detector_hits["energy"][count])
                energies2.append(self.detector_hits["energy"][count + 1])

                # Successful pair, continues to next possible pair
                count = count + 2
                array_counter = array_counter + 1
            else:
                # Unsuccessful pair, continues
                count = count + 1

        hits_pos1 = [(X1, Y1) for X1, Y1 in zip(posX1, posY1)]
        hits_pos2 = [(X2, Y2) for X2, Y2 in zip(posX2, posY2)]

        hits_dict = {
            "Position1": hits_pos1,
            "Position2": hits_pos2,
            "Energy1": energies1,
            "Energy2": energies2,
        }

        self.hits_dict = hits_dict

        print("processed detector hits")

        return hits_dict

    # get hits for generalized setup in geant4
    def get_det_hits(
        self,
        remove_secondaries: bool = False,
        second_axis: str = "y",
        energy_level: float = 500,
    ) -> dict:
        """
        return a dictionary containing hits on front detector

        params:
        returns:
            hits_dict: contains positions of hits for front detector in particle order, also has energies
        """

        posX = []
        posY = []
        energies = []
        vertex = []
        detector_offset = 1111 * 0.45 - (0.03 / 2)  # TODO: make this dynamic

        secondary_e = 0
        secondary_gamma = 0

        for count, el in enumerate(self.detector_hits["det"]):
            # only get hits on the first detector
            if el == 1 and remove_secondaries != True:
                xpos = self.detector_hits["x"][count]
                ypos = self.detector_hits["y"][count]
                zpos = self.detector_hits["z"][count]
                energy_keV = self.detector_hits["energy"][count]

                if (
                    second_axis == "z" and ypos == 0.015
                ):
                    posX.append(xpos)
                    posY.append(zpos - detector_offset)
                    energies.append(energy_keV)
                    vertex.append(
                        np.array(
                            [
                                self.detector_hits["x0"][count],
                                self.detector_hits["y0"][count],
                                self.detector_hits["z0"][count],
                            ]
                        )
                    )
                    # add to secondary count -- anything NOT parent
                    if self.detector_hits["ID"][count] != 0:
                        if self.detector_hits["name"][count] == "e-":
                            secondary_e += 1
                        else:
                            secondary_gamma += 1
                elif (
                    second_axis == "y"
                    and zpos == detector_offset):
                    posX.append(xpos)
                    posY.append(ypos)
                    energies.append(energy_keV)
                    vertex.append(
                        np.array(
                            [
                                self.detector_hits["x0"][count],
                                self.detector_hits["y0"][count],
                                self.detector_hits["z0"][count],
                            ]
                        )
                    )
                     # add to secondary count -- anything NOT parent
                    if self.detector_hits["ID"][count] != 0:
                        if self.detector_hits["name"][count] == "e-":
                            secondary_e += 1
                        else:
                            secondary_gamma += 1
                else:
                    pass

            # if checking for secondaries and want to remove them, only process electrons
            elif el == 1 and remove_secondaries:
                if (
                    self.detector_hits["ID"][count] == 0
                    and self.detector_hits["name"][count] == "e-"
                ):
                    xpos = self.detector_hits["x"][count]
                    ypos = self.detector_hits["y"][count]
                    zpos = self.detector_hits["z"][count]
                    energy_keV = self.detector_hits["energy"][count]

                    if (
                        second_axis == "z"
                        and ypos == 0.015
                        and energy_level - energy_level * 0.05
                        < energy_keV
                        < energy_level * 0.05 + energy_level
                    ):
                        posX.append(xpos)
                        posY.append(zpos - detector_offset)
                        energies.append(energy_keV)
                        vertex.append(
                            np.array(
                                [
                                    self.detector_hits["x0"][count],
                                    self.detector_hits["y0"][count],
                                    self.detector_hits["z0"][count],
                                ]
                            )
                        )
                    elif (
                        second_axis == "y"
                        and zpos == detector_offset
                        and energy_level - energy_level * 0.01
                        < energy_keV
                        < energy_level * 0.01 + energy_level
                    ):
                        posX.append(xpos)
                        posY.append(ypos)
                        energies.append(energy_keV)
                        vertex.append(
                            np.array(
                                [
                                    self.detector_hits["x0"][count],
                                    self.detector_hits["y0"][count],
                                    self.detector_hits["z0"][count],
                                ]
                            )
                        )
                    else:
                        pass
            else:
                pass

        hits_pos = [(X, Y) for X, Y in zip(posX, posY)]

        hits_dict = {"Position": hits_pos, "Energy": energies, "Vertices": vertex}

        self.hits_dict = hits_dict

        print("processed detector hits")

        return hits_dict, secondary_e, secondary_gamma

    # get hits simulated in experiment set up in geant4
    def get_experiment_geant4_hits(self) -> dict:
        """
        return a dictionary containing hits on Minipix EDU from simulated detector in geant4

        params:
        returns:
            hits_dict: contains positions of hits, also has energies
        """

        posX = []
        posY = []
        energies = []
        for count in range(len(self.detector_hits["energy"]) - 1):
            xpos = self.detector_hits["x"][count]
            zpos = self.detector_hits["z"][count]
            energy_kev = self.detector_hits["energy"][count]

            # save
            posX.append(xpos)
            posY.append(zpos)
            energies.append(energy_kev)

        hits_pos = [(X, Y) for X, Y in zip(posX, posY)]

        hits_dict = {"Position": hits_pos, "Energy": energies}

        self.hits_dict = hits_dict

        print("processed detector hits")

        return hits_dict

    def update_pos_uncertainty(
        self, det_size_cm: float, dist_type: str, dist_param: float
    ) -> None:
        """
        return a dictionary containing hits on both detectors

        params:
            dist_type:  "Gaussian", "Poission", or "Uniform" distribution to sample uncertainty from
            dist_param: determined by dist_type, in milimeters
                        if "Gaussian" standard deviation
                        if "Poission" 1/lambda
                        if "Uniform" bounds for uniform
        returns:
        """

        if len(self.hits_dict.keys()) > 2:
            # contains two positions entries
            pos1 = self.hits_dict["Position1"]
            pos2 = self.hits_dict["Position2"]

            unc_pos1 = [
                add_uncertainty(pos, dist_type, dist_param, det_size_cm) for pos in pos1
            ]
            unc_pos2 = [
                add_uncertainty(pos, dist_type, dist_param, det_size_cm) for pos in pos2
            ]

            # update entries
            self.hits_dict["Position1"] = unc_pos1
            self.hits_dict["Position2"] = unc_pos2

        else:
            # contains one position entry
            pos_p = self.hits_dict["Position"]

            unc_pos = [
                add_uncertainty(pos, dist_type, dist_param, self.det_size_cm)
                for pos in pos_p
            ]

            # unpdate entry
            self.hits_dict["Position"] = unc_pos

        return

    def exclude_pcfov(
        self,
        detector_dim,
        mask_dim,
        focal_length,
        plane_distance,
        second_axis,
        sphere_radius,
    ):
        # calculate anlge between origin and the hit on the detector
        count = 0
        inside_idx = []

        import matplotlib.pyplot as plt

        plot = False

        def ray_sphere_intersection(
            ray_origin, ray_direction, sphere_center, sphere_radius
        ):
            # Calculate the vector from the ray's origin to the sphere's center
            oc = ray_origin - sphere_center

            # Calculate coefficients for the quadratic equation
            a = np.dot(ray_direction, ray_direction)
            b = 2.0 * np.dot(oc, ray_direction)
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

        # center is at 0,0,0 where the imaginary plane is
        mask0 = np.array(
            [mask_dim / 2, mask_dim / 2, -1 * (plane_distance + focal_length)]
        )
        mask1 = np.array(
            [-1 * mask_dim / 2, mask_dim / 2, -1 * (plane_distance + focal_length)]
        )
        mask2 = np.array(
            [mask_dim / 2, -1 * mask_dim / 2, -1 * (plane_distance + focal_length)]
        )
        mask3 = np.array(
            [-1 * mask_dim / 2, -1 * mask_dim / 2, -1 * (plane_distance + focal_length)]
        )
        sphere_center = np.array([0, 0, 0])

        # Calculate the vector from point1 to point2
        vector0 = mask0 - sphere_center
        vector1 = mask1 - sphere_center
        vector2 = mask2 - sphere_center
        vector3 = mask3 - sphere_center

        ray_directions = [vector0, vector1, vector2, vector3]

        intersection_points = []
        for vector in ray_directions:
            # ray origin is from sphere center
            ray_origin = sphere_center
            vector /= np.linalg.norm(vector)

            # Calculate the intersection point on the surface of the sphere
            intersection_point = ray_sphere_intersection(
                ray_origin, vector, sphere_center, sphere_radius
            )
            intersection_points.append(intersection_point)

        # now we go through each input and check if its inside the square
        x1 = min([itp[0] for itp in intersection_points])
        x2 = max([itp[0] for itp in intersection_points])
        y1 = min([itp[1] for itp in intersection_points])
        y2 = max([itp[1] for itp in intersection_points])

        for vi, vtx in enumerate(self.hits_dict["Vertices"]):
            if (x1 < vtx[0] and vtx[0] < x2) and (y1 < vtx[1] and vtx[1] < y2):
                inside_idx.append(vi)
                if plot:
                    plt.scatter(vtx[0], vtx[1], color="blue")
            else:
                count += 1
                if plot:
                    plt.scatter(vtx[0], vtx[1], color="red")

        inside_hits = {}
        # save only those inside
        for key, array in self.hits_dict.items():
            inside_hits[key] = [array[i] for i in inside_idx]

        self.hits_dict = inside_hits
        if plot:
            plt.show()

        print("removed ", count, "hits")
        """


        focal_length = focal_length*-1
        # create locations for sensor corners and mask corners

        if second_axis == "z":
            pixel0 = np.array([detector_dim / 2, -1 * focal_length, detector_dim / 2])
            mask0 = np.array([mask_dim / 2, 0, mask_dim / 2])

            pixel1 = np.array(
                [-1 * detector_dim / 2, -1 * focal_length, detector_dim / 2]
            )
            mask1 = np.array([-1 * mask_dim / 2, 0, mask_dim / 2])

            pixel2 = np.array(
                [detector_dim / 2, -1 * focal_length, -1 * detector_dim / 2]
            )
            mask2 = np.array([mask_dim / 2, 0, -1 * mask_dim / 2])

            pixel3 = np.array(
                [-1 * detector_dim / 2, -1 * focal_length, -1 * detector_dim / 2]
            )
            mask3 = np.array([-1 * mask_dim / 2, 0, -1 * mask_dim / 2])
        else:
            pixel0 = np.array(
                [
                    detector_dim / 2,
                    detector_dim / 2,
                    -1 * focal_length,
                ]
            )
            mask0 = np.array([mask_dim / 2, mask_dim / 2, 0])

            pixel1 = np.array([-1 * detector_dim / 2, detector_dim / 2, -1 * focal_length])
            mask1 = np.array([-1 * mask_dim / 2, mask_dim / 2, 0])

            pixel2 = np.array([detector_dim / 2, -1 * detector_dim / 2, -1 * focal_length])
            mask2 = np.array([mask_dim / 2, -1 * mask_dim / 2, 0])

            pixel3 = np.array([-1 * detector_dim / 2, -1 * detector_dim / 2, -1 * focal_length])
            mask3 = np.array([-1 * mask_dim / 2, -1 * mask_dim / 2, 0])
        # Calculate the vector from point1 to point2
        vector0 = mask0 - pixel0
        vector1 = mask1 - pixel1
        vector2 = mask2 - pixel2
        vector3 = mask3 - pixel3

        ray_directions = [vector0, vector1, vector2, vector3]


        def ray_plane_intersection(ray_origin, ray_direction, plane_distance):
            P0 = np.array([0, 0, -1 * plane_distance])
            N = np.array([0, 0, 1])

            dot_product = np.dot(ray_direction, N)
            # Check if the vector is parallel to the plane
            if abs(dot_product) < 1e-6:
                print("Vector is parallel to the plane, no intersection.")
                return
            else:
                # Calculate the parameter t at which the vector intersects the plane
                t = np.dot(P0 - ray_origin, N) / dot_product

                # Calculate the intersection point
                intersection_point = ray_origin + t * ray_direction
            return intersection_point

        # Define the ray's origin and direction
        pixels = [pixel0, pixel1, pixel2, pixel3]
        vectors = [vector0, vector1, vector2, vector3]


        
        # check if the point lies on the 2D polygon formed
        polygon_coords = [(itp[0], itp[1]) for itp in intersection_points]
        polygon_coords.append((intersection_points[0][0], intersection_points[0][1]))
        polygon_coords[2], polygon_coords[3] = polygon_coords[3], polygon_coords[2]
        src_polygon = Polygon(polygon_coords)

        inside_idx = []
        for vi, vtx in enumerate(self.hits_dict["Vertices"]):
            point = Point(vtx[0], vtx[1])
            if src_polygon.contains(point):
                inside_idx.append(vi)
            else:
                # print(point)
                pass

        inside_hits = {}
        # save only those inside
        for key, array in self.hits_dict.items():
            inside_hits[key] = [array[i] for i in inside_idx]

        self.hits_dict = inside_hits
        
        # find the theta and phi extent from the intersection points
        theta_extents = []
        phi_extents = []
        # plotting
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        for itp in intersection_points:
            x = itp[0]
            y = itp[1]
            z = itp[2]

            r = math.sqrt(x**2 + y**2 + z**2)
            phi = math.atan2(y, x)
            theta = math.acos(z / r)

            theta_extents.append(theta)
            phi_extents.append(phi)
            ax.scatter(x,y,z,color='black')

        # unpack into extents
        theta_min = min(np.array(theta_extents))
        theta_max = max(np.array(theta_extents))
        phi_min = min(np.array(phi_extents))
        phi_max = max(np.array(phi_extents))
        print(theta_max)

        inside_idx = []


        radius = 7
        u = np.linspace(0, 2 * np.pi, 1000)
        v = np.linspace(0, np.pi, 1000)
        x = radius * np.outer(np.cos(u), np.sin(v))
        y = radius * np.outer(np.sin(u), np.sin(v))
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='c', alpha=0.3)
        origin = np.zeros(3)
        scale = 10

        for px,vector in zip(pixels,vectors):
            vector /= np.linalg.norm(vector)
            ax.plot([px[0],scale*vector[0]], [px[1],scale*vector[1]], [px[2],scale*vector[2]])

        # finally, go through hits and determine if within extents
        theta_max = 0.583373
        for vi, vtx in enumerate(self.hits_dict["Vertices"]):
            # adjust for sphere center - TODO fix this to be dynamic!
            xc = 0
            yc = 0
            zc = 497.915
            x = vtx[0] - xc
            y = vtx[1] - yc
            z = vtx[2] - zc
            
            # calculate in spherical
            r = math.sqrt(x**2 + y**2 + z**2)
            phi = math.atan2(y, x)
            theta = np.pi - math.acos(z / r)

            if second_axis == "z":
                if theta < theta_max:
                    inside_idx.append(vi)
                else:
                    pass
            else:
                if theta < theta_max:
                    inside_idx.append(vi)
                    ax.scatter(x,y,z,color="blue")

                else:
                    ax.scatter(x,y,z,color="red")
                    #pass
        inside_hits = {}
        # save only those inside
        for key, array in self.hits_dict.items():
            inside_hits[key] = [array[i] for i in inside_idx]

        self.hits_dict = inside_hits

        #plt.show()
        """
        return

    def calc_angle(self):
        detector_offset = 1111 * 0.45 - (0.03 / 2)  # TODO: make this dynamic
        angles = []
        for vi, (vtx, ptx) in enumerate(
            zip(self.hits_dict["Vertices"], self.hits_dict["Position"])
        ):
            v_x = ptx[0] - vtx[0]
            v_y = ptx[1] - vtx[1]
            v_z = detector_offset - vtx[2]

            angle = math.atan(abs(v_z) / math.sqrt(v_x**2 + v_y**2))

            # Convert the angle from radians to degrees
            angle_degrees = 90 - math.degrees(angle)
            angles.append(angle_degrees)
        mean_angle = np.mean(np.array(angles))
        std = np.std(np.array(angles))

        print("SOURCE RUN AT ", mean_angle, " WITH SPREAD", std)
        return mean_angle, std
