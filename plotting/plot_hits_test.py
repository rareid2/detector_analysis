import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

# Read the CSV file into a DataFrame
file_path = "/home/rileyannereid/workspace/geant4/simulation-data/sine-test/11-None-0_1.00E+08_Mono_500.csv"
data = pd.read_csv(file_path, header=None)

# Assuming columns 2 and 4 are the 2nd and 4th columns (index 1 and 3)
column2_data = data.iloc[:, 1]
column3_data = data.iloc[:, 2]
column4_data = data.iloc[:, 3]

xes = []
yes = []

for x,y,z in zip(column2_data,column3_data, column4_data):
    if z == 499.935:
        xes.append(x)
        yes.append(y)



# Create a 2D histogram using numpy
hist2d, x_edges, y_edges = np.histogram2d(
    xes, yes, bins=22
)  # You can adjust the number of bins

# Plot the heatmap using seaborn
plt.imshow(hist2d)
plt.show()
plt.clf()

# Read coordinates from the text file
with open('/home/rileyannereid/workspace/geant4/EPAD_geant4/src/mask_designs/11mosaicMURA_matrix_25.41.txt', 'r') as file:
    lines = file.readlines()


coordinates = [line.strip().split(',') for line in lines]
x_coords = [float(coord[0]) for coord in coordinates]
y_coords = [float(coord[1]) for coord in coordinates]

# Define the side length of the squares
side_length = 1.21  # Adjust this value based on your data

# Create a plot
plt.figure(figsize=(8, 8))  # Adjust the figure size as needed

for x, y in zip(x_coords, y_coords):
    # Calculate the coordinates of the square's vertices
    vertices = [
        (x, y),  # Lower left
        (x + side_length, y),  # Lower right
        (x + side_length, y + side_length),  # Upper right
        (x, y + side_length)  # Upper left
    ]
    
    # Create a Polygon patch and add it to the plot
    square = Polygon(vertices, closed=True, facecolor='blue', alpha=0.5)
    plt.gca().add_patch(square)

# Set plot labels and title
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Squares Plot with Shading')

# Set axis limits based on the data
plt.xlim(min(x_coords) - 1, max(x_coords) + side_length + 1)
plt.ylim(min(y_coords) - 1, max(y_coords) + side_length + 1)

# Show the plot
#plt.show()