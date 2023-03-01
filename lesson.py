import numpy as np
import matplotlib.pyplot as plt


# Parameters
num_beads = 1
bead_diameter = 10  # microns
spot_diameter = 1  # microns
threshold = 0.5
max_iterations = 100

# Generate beads randomly
beads_x = np.random.uniform(0, 100, num_beads)
beads_y = np.random.uniform(0, 100, num_beads)

# Create image grid
grid_size = 100  # microns
grid_step = 0.5  # microns
x, y = np.meshgrid(np.arange(0, grid_size, grid_step), np.arange(0, grid_size, grid_step))

# Generate fluorescence from beads as Gaussian functions
z = np.zeros_like(x)
for bead_x, bead_y in zip(beads_x, beads_y):
    bead_z = np.exp(-((x - bead_x)**2 + (y - bead_y)**2) / (2 * (bead_diameter/2)**2))
    z += bead_z

# Display initial image
fig, ax = plt.subplots()
im = ax.imshow(z, cmap='hot', extent=(0, grid_size, 0, grid_size))
ax.set_xlabel('x (microns)')
ax.set_ylabel('y (microns)')
ax.set_title('Fluorescent beads')

# Start with a random spot
x0, y0 = np.random.uniform(0, grid_size, 2)
trajectory = [(x0, y0)]
order = [0]
for i in range(max_iterations):
    # Create spot as Gaussian function
    spot = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * (spot_diameter/2)**2))

    # Acquire fluorescence signal
    signal = np.sum(z * spot)

    # Check if signal exceeds threshold
    if signal > threshold:
        # Calculate gradient and move to next spot in direction of gradient
        dx = np.sum((x - x0) * z * spot) / signal
        dy = np.sum((y - y0) * z * spot) / signal
        x0 += dx
        y0 += dy
        trajectory.append((x0, y0))
        order.append(i+1)
    else:
        # Start over with a new random spot
        x0, y0 = np.random.uniform(0, grid_size, 2)
        trajectory.append((x0, y0))
        order.append(i+1)

# Plot trajectory with lines between spots
fig, ax = plt.subplots()
im = ax.imshow(z, cmap='hot', extent=(0, grid_size, 0, grid_size))
ax.set_xlabel('x (microns)')
ax.set_ylabel('y (microns)')
ax.set_title('Trajectory of spots')
for i in range(len(trajectory)-1):
    x1, y1 = trajectory[i]
    x2, y2 = trajectory[i+1]
    ax.plot([x1, x2], [y1, y2], color='white', alpha=0.5, linewidth=2)
    ax.plot([x1], [y1], marker='o', markersize=4, color='blue')
ax.plot([x2], [y2], marker='o', markersize=4, color='red', label='Final spot')
ax.legend()
plt.show()
