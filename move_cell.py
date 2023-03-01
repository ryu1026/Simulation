import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# define the space
space_size = 100  # size of the space in µm
grid_step = 1  # distance between each grid point in µm
num_circles = 10  # number of circles to create
circle_diameter = 10  # diameter of each circle in µm

# create a figure and axis object
fig, ax = plt.subplots()

# create the initial plot with empty circles
circles = []
for i in range(num_circles):
    circle = plt.Circle((0, 0), circle_diameter / 2, fill=False)
    circles.append(circle)
    ax.add_artist(circle)


# function to update the position of the circles at each frame
def update(frame):
    # clear the axis before plotting
    ax.clear()

    # create the new circle positions
    circle_positions = np.random.uniform(0, space_size, size=(num_circles, 2))

    # update the positions of the circles
    for i, circle in enumerate(circles):
        circle.center = circle_positions[i]
        ax.add_artist(circle)

    # set the axis limits and labels
    ax.set_xlim(0, space_size)
    ax.set_ylim(0, space_size)
    ax.set_xlabel('X position (µm)')
    ax.set_ylabel('Y position (µm)')
    ax.set_title('Moving circles')


# create the animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 100), interval=200, repeat=True)

# show the plot
plt.show()
