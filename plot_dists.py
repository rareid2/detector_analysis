import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


materials = ["Si", "Be"]
sigmas = []
for material in materials:
    charge_nmbr = 1
    rest_mass_MeV = 0.511
    det1_thickness_um = 100
    energy_keV = 500

    if material == "Be":
        X0 = 352.7597513557388 * 10**3  # in um
        Z_det = 4
        material = material
    elif material == "Si":
        Z_det = 14
        X0 = 37 * 10**3  # in um
        material = material
    else:
        print("choose Be or Si")

    # charge number
    z = charge_nmbr
    # thickness
    x = det1_thickness_um

    # print('characteristic length is:', x/X0)
    if x / X0 > 100:
        print(x, energy_keV)

    # updated beta_cp (from src code)
    E = energy_keV * 0.001  # convert to ME
    invbeta_cp = (E + rest_mass_MeV) / (E**2 + 2 * rest_mass_MeV * E)

    # term 1 and 2 for the distribution -- urban 2006 eqn 28
    fz = 1 - (0.24 / (Z_det * (Z_det + 1)))
    t1 = 13.6 * invbeta_cp  # in MeV
    t2 = (
        z
        * np.sqrt(x / X0)
        * np.sqrt(1 + 0.105 * np.log(x / X0) + 0.0035 * (np.log(x / X0)) ** 2)
        * fz
    )

    # compute theoretical sigma for normal distribution
    sigma = np.rad2deg(t1 * t2)

    mu = 0
    sigmas.append(sigma)
print(sigmas)
fig = plt.figure()


plt.rcParams.update({"font.family": "calibri"})
colors = ["#34C0D2", "#7F7F7F"]
for cc, sigma in zip(colors, sigmas):
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    plt.plot(x, scipy.stats.norm.pdf(x, mu, sigma), color=cc)
    plt.fill_between(x, scipy.stats.norm.pdf(x, mu, sigma), color=cc, alpha=0.5)

fig.savefig("temp.png", transparent=True, dpi=500)

fig.savefig("temp.png", transparent=False, dpi=500)
