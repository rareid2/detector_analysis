import numpy as np
import matplotlib.pyplot as plt

# Number of random points
num_points = 1000

# Generate random angles in radians
theta = np.random.uniform(0, 2 * np.pi, num_points)

s = 1

r = np.random.uniform(0, np.sqrt(2)*s/2, num_points)

# Convert polar coordinates to Cartesian coordinates
x = r * np.cos(theta)
y = r * np.sin(theta)

for i, (th, rr) in enumerate(zip(theta, r)):
    if  (np.absolute(rr*np.cos(th)-rr*np.sin(th))+ np.absolute(rr*np.cos(th)+rr*np.sin(th))) <= s:
        plt.scatter(x[i], y[i], color='blue')  # 'bo' represents blue circles
    else:
        plt.scatter(x[i], y[i], color='red')  # 'bo' represents blue circles

plt.title('Random Points in Polar Coordinates')
plt.show()