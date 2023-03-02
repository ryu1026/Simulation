import numpy as np
kb = 1.380649e-23  # Boltzmann constant in J/K
# commit test
# commit_test2
# Define simulation parameters
signal_duration = 1e-6  # Signal duration in seconds
sampling_rate = 1e9     # Sampling rate in Hz
amplifier_bandwidth = 100e6  # Amplifier bandwidth in Hz
noise_figure = 2.0      # Amplifier noise figure in dB

# Define time vector
t = np.arange(0, signal_duration, 1 / sampling_rate)

# Define input signal
x = np.random.poisson(lam=1, size=len(t))  # Poisson-distributed photon flux

# Define low-pass filter
rc = 1 / (2 * np.pi * amplifier_bandwidth)
alpha = 1 / (rc * sampling_rate + 1)
lpf = [alpha]
for n in range(1, len(t)):
    lpf.append(alpha * (1 - alpha) ** n)

# Apply low-pass filter
y = np.convolve(x, lpf)[:len(t)]

# Calculate amplifier noise
amplifier_noise = np.random.normal(scale=np.sqrt(kb * T * 10 ** (noise_figure / 10)), size=len(t))

# Amplify signal and add noise
gain = 1e6
v = gain * y + amplifier_noise

# Check if v is empty
if np.all(v == 0):
    v = np.zeros_like(t)

# Plot results
import matplotlib.pyplot as plt

fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
ax[0].plot(t, x)
ax[0].set_ylabel('Photon flux')
ax[1].plot(t, y)
ax[1].set_ylabel('Signal')
ax[2].plot(t, v)
ax[2].set_ylabel('Amplified signal')
ax[2].set_xlabel('Time (s)')
plt.show()
