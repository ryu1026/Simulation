import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
signal_duration = 1e-6  # Signal duration in seconds
sampling_rate = 1e9     # Sampling rate in Hz
excitation_wavelength = 488e-9  # Excitation wavelength in meters
emission_wavelength = 525e-9   # Emission wavelength in meters
quantum_yield = 0.5            # Quantum yield of fluorescent beads
background_intensity = 100     # Background intensity in counts per second
photoelectrons_count = 50       # Average number of photoelectrons per photon
h = 6.626e-34      # Planck constant in J s
c = 299792458      # Speed of light in m/s

# Define time vector
t = np.arange(0, signal_duration, 1 / sampling_rate)

# Simulate photon flux
photon_flux = np.random.poisson(lam=1e8, size=len(t))  # Poisson-distributed photon flux

# Calculate fluorescence emission
emission_spectrum = np.zeros_like(t)
for i in range(len(t)):
    if photon_flux[i] > 0:
        photon_energy = h * c / excitation_wavelength  # Energy of one photon in joules
        emitted_photons = quantum_yield * photon_flux[i]  # Number of emitted photons
        emitted_energy = emitted_photons * photon_energy  # Energy of emitted photons in joules
        emitted_power = emitted_energy / signal_duration  # Average emitted power in watts
        emitted_photons_per_sec = emitted_photons / signal_duration  # Average number of emitted photons per second
        emitted_photons_per_sample = emitted_photons_per_sec / sampling_rate  # Average number of emitted photons per sample
        emitted_photons_per_sample = np.random.poisson(lam=emitted_photons_per_sample)  # Add poisson noise
        emission_spectrum[i] = emitted_photons_per_sample * photoelectrons_count

# Add background intensity
background_counts = background_intensity / sampling_rate
emission_spectrum += np.random.poisson(lam=background_counts, size=len(t))

# Plot results
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
ax[0].plot(t, photon_flux)
ax[0].set_ylabel('Photon flux')
ax[1].plot(t, emission_spectrum)
ax[1].set_ylabel('Counts')
ax[1].set_xlabel('Time (s)')
plt.show()
