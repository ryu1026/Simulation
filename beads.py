import numpy as np
import matplotlib.pyplot as plt

# Set the size of the 2D plane
size = 100

# Set the number of steps and the step size
num_steps = 1000
step_size = 0.1

# Initialize the starting position to the center of the plane
x = np.array([size/2])
y = np.array([size/2])

# Generate the random steps
dx = np.random.normal(scale=step_size, size=num_steps)
dy = np.random.normal(scale=step_size, size=num_steps)

# Compute the cumulative sum of the steps to get the random walk path
x = np.concatenate([x, np.cumsum(dx)])
y = np.concatenate([y, np.cumsum(dy)])

# Clip the path to ensure it stays within the boundaries of the plane
x = np.clip(x, 0, size)
y = np.clip(y, 0, size)

# Plot the random walk path with numbered points
fig, ax = plt.subplots()
ax.plot(x, y)
for i in range(num_steps):
    ax.text(x[i], y[i], i+1)

# Add a title and axis labels
plt.title('Random Walk in 2D Plane')
plt.xlabel('X')
plt.ylabel('Y')

# Show the plot
plt.show()
