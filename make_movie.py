import numpy as np
import matplotlib.pyplot as plt
import imageio

ns = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
images = []
for n in ns:
    if n == 0:
        fname = (
            f"../simulation-results/19-1-fwhm/19-2.4-{n}-0-0_1.00E+06_Mono_500_dc.png"
        )
    else:
        fname = (
            f"../simulation-results/19-1-fwhm/19-2.4-{n}-xy-0_1.00E+06_Mono_500_dc.png"
        )
    print(fname)
    images.append(imageio.imread(fname))

imageio.mimsave("../simulation-results/19-1-fwhm/fwhm.gif", images, duration=1)
