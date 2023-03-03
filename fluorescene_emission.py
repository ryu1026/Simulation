import numpy as np
import matplotlib.pyplot as plt

# Set simulation parameters
n_beads = 10000   # number of beads
d_beads = 10e-6   # bead diameter (m)
q_yield = 0.9     # fluorescence quantum yield
excitation_wl = 488e-9  # excitation wavelength (m)
excitation_power = 1e-3  # excitation power (W)
NA = 0.2          # numerical aperture
refractive_index = 1.33  # refractive index of medium

# Calculate excitation light intensity at each bead location
r_beads = d_beads / 2
bead_locs = np.random.uniform(low=-r_beads, high=r_beads, size=(n_beads, 3))
distances = np.sqrt(np.sum(np.square(bead_locs), axis=1))
intensities = excitation_power / (4 * np.pi * distances**2) * np.exp(-distances / (2 * r_beads))

# Calculate fluorescence emission at each bead location
emission_wl = 510e-9   # emission wavelength (m)
h = 6.626e-34  # Planck's constant (J s)
c = 2.998e8    # speed of light (m/s)
energy_photon = h * c / emission_wl  # energy of emitted photon (J)
emission_rate = q_yield * intensities / energy_photon
emitted_photons = np.random.poisson(emission_rate)

# Plot histogram of emitted photon counts
plt.hist(emitted_photons, bins=50)
plt.xlabel('Number of emitted photons')
plt.ylabel('Counts')
plt.show()
