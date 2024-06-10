import numpy as np
from scipy.integrate import simps, trapz
from scipy.interpolate import interp1d

data = np.loadtxt("e-spectrum.txt", delimiter=",")

# Split the data into X and Y arrays
X = data[:, 0]
Y = data[:, 1]

f = interp1d(X, Y, kind="linear")
x_new = np.logspace(np.log10(min(X)), np.log10(max(X)), num=1000)
y_new = f(x_new)

data_x = x_new
data_y = y_new

X1 = 300
X2 = 3000

# Interpolate to get the indices corresponding to X1 and X2
indices = np.where((data_x >= X1) & (data_x <= X2))

# Select the subarrays for the integration limits
X_sub = data_x[indices]
Y_sub = data_y[indices]

# Perform the integration using Simpson's rule
integral_simps = simps(Y_sub, X_sub)

# Alternatively, use the trapezoidal rule
integral_trapz = trapz(Y_sub, X_sub)

print(f"Integral between {X1} and {X2} using Simpson's rule: {integral_simps}")
print(f"Integral between {X1} and {X2} using trapezoidal rule: {integral_trapz}")
