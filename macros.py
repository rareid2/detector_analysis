# functions to create each type of macro file

def create_angle_beam_macro(n_particles, energy_in_keV, position, direction):
    macro_path = "../EPAD_geant4/macros/run_angle_beam.mac"
    with open(macro_path, "w") as f:
        f.write("/run/numberOfThreads 40 \n")
        f.write("/run/initialize \n")
        f.write("/control/verbose 0 \n")
        f.write("/run/verbose 0 \n")
        f.write("/event/verbose 0 \n")
        f.write("/tracking/verbose 0 \n")

        f.write("/gps/particle e- \n")
        f.write("/gps/energy " + str(energy_in_keV) + " keV \n")
        f.write(
            "/gps/position "
            + str(position[0])
            + " "
            + str(position[1])
            + " "
            + str(position[2])
            + " cm\n"
        )
        f.write(
            "/gps/direction "
            + str(direction[0])
            + " "
            + str(direction[1])
            + " "
            + str(direction[2])
            + " \n"
        )
        f.write("/gps/pos/type Point \n")

        f.write("/run/beamOn " + str(n_particles) + " \n")
        f.close()
