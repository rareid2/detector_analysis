import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


log_data = [
    [80.4174421047028, 301936.8199781656],
    [336.4827440441997, 15711.353238960946],
    [479.77509558648046, 3374.73797132462],
    [535.6166267348216, 2010.7643982229145],
    [736.1904348514545, 387.92351710604953],
    [802.0116270028464, 239.25397531517257],
    [1024.3284110984293, 56.13042416368999],
    [1115.9113955834737, 32.30929848409303],
    [1274.0498745626332, 10.56223301682826],
    [1200.9012001156673, 20.39084384745854],
    [1478.5194766501274, 5.00914488422795],
    [1591.1261601331228, 3.019131509569519],
    [1712.3091697080604, 1.791997462989937],
    [1842.7216936845512, 1.080079760243438],
    [1958.954672822181, 0.7247960349158931],
    [154.69407652461987, 128502.2970787311],
    [204.4213800044951, 71037.08771708737],
    [257.351270001691, 35847.21705777606],
    [407.87427589690174, 6943.591429214298],
    [907.6005216818141, 122.77470827878692],
]

x_data = [ld[0] for ld in log_data]
y_data = [ld[1] * 0.25 * ld[0] for ld in log_data]


def func_log(x, a, b):
    return x**a + b


# Fit the curve to the log-scaled data
params, covariance = curve_fit(func_log, x_data, y_data)
# Print the fitted parameters
print("Fitted parameters:", params)

# Plot the original log-scaled data and the fitted curve
plt.scatter(x_data, y_data, label="Original Log-scaled Data")
# New x values for interpolation
x_interp_log = np.logspace(1, 3.3, 1000)
from scipy.interpolate import interp1d

# Create an interpolation function with linear interpolation
interp_func = interp1d(x_data, y_data, kind="linear", fill_value="extrapolate")
y_interp_log = interp_func(x_interp_log)
plt.loglog(x_interp_log, y_interp_log, color="red", label="Interpolated Curve")
plt.legend()
plt.show()

# kk here we go
bin_edges = np.logspace(2, 3, 24)
bin_centers = []
for bi, be in enumerate(bin_edges[:-1]):
    bc = (bin_edges[bi + 1] - be) / 2 + be
    bin_centers.append(bc)
    print((bin_edges[bi + 1] - be) / bc)

# so we have to run each of these energies but pretend the above bins

# THEN we need to do the same for pitch angles

pitch_angle_data = [
    [4.0316625474712, 6310036.260811816],
    [7.570905419253705, 5123017.6427039],
    [10.949273615046103, 3899698.193624],
    [13.845017782868148, 2889552.074698],
    [16.419012598709976, 2198565.900710],
    [18.993007414551798, 1599654.825088],
    [21.406127554403504, 1200251.444176],
    [23.497498342274987, 915227.1626104],
    [26.07149315811681, 673502.43993565],
    [29.128112001928976, 487371.5094179],
    [32.02385616975103, 363059.27677150],
    [35.24134968955331, 276642.33868300],
    [38.780592561335816, 210456.4377596],
    [42.31983543311833, 171484.81923490],
    [45.859078304900834, 146627.4859799],
    [49.39832117668334, 129362.12920077],
    [52.937564048465845, 119331.6641433],
    [56.47680692024835, 110553.54005535],
    [60.016049792030856, 106044.0730941],
    [63.55529266381336, 104145.15221295],
    [67.09453553559587, 103146.33716779],
    [70.63377840737837, 102280.23512289],
    [74.17302127916088, 101299.30576140],
    [77.7122641509434, 101299.305761406],
    [81.2515070227259, 101299.305761406],
    [84.79074989450841, 101299.30576140],
    [88.32999276629091, 101299.30576140],
    [91.86923563807342, 102280.23512289],
    [95.40847850985593, 103146.33716779],
    [98.94772138163843, 102898.13376672],
    [102.48696425342094, 103395.1392670],
    [106.02620712520346, 103519.7652797],
    [109.56544999698596, 104019.7733056],
    [113.10469286876847, 105788.8968045],
    [116.64393574055097, 105576.7190149],
    [120.18317861233348, 105069.2269758],
    [123.72242148411598, 107251.6710830],
    [127.26166435589849, 107562.1915432],
    [130.800907227681, 106325.477992795],
    [134.3401500994635, 111654.82134811],
    [137.879392971246, 109549.806501479],
    [155.57560733015856, 117950.1565397],
    [166.19333594550608, 116210.7390635],
    [177.9371872927844, 120209.34479128],
    [141.37380191693293, 112571.5306862],
    [146.80511182108629, 115573.2012329],
    [151.4376996805112, 117103.91877237],
    [160.22364217252397, 118654.9100098],
    [169.6485623003195, 120226.44346174],
    [174.2811501597444, 121818.79120101],
]

de_E = 0.17  # For Cassini CAPS ELS
central_energy = 235

# so just simulate at 235
# so for each bin, we need to simulate the flux given on the left but we also need GF

# what design should I simulate? to get coverage over 50 deg???? maybe do

f_number = 0.4
# rank in the 40s?
