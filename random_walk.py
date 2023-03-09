import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.use('TkAgg')


# Set up the figure and axis for plotting
fig, ax = plt.subplots()
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

# Initialize the plot with a single point at the origin
point, = ax.plot(0, 0, 'bo')


def random_walk(x_pre, y_pre):
    # 直前の集光点の座標(x0, y0)を受け取って次の集光点(x,y)をランダムに与える
    x_pre += np.random.normal(scale=10)  # 平均0，標準偏差: step_sizeの乱数
    y_pre += np.random.normal(scale=10)
    x_next = np.clip(x_pre, 0, 100)  # 最小値が0で最大値がx_grid_sizeに指定される
    y_next = np.clip(y_pre, 0, 100)

    return x_next, y_next


# Function to update the plot with the next point in the random walk
def update():
    global x, y
    x, y = random_walk(x, y)
    point.set_data(x, y)
    return point,


# Generate the random walk points
x, y = 50, 50
n_points = 100
points = [(x, y)]
for i in range(n_points):
    x, y = random_walk(x, y)
    points.append((x, y))

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=n_points, interval=100, blit=True)

# Show the animation
plt.show()
