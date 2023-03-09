import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set up the figure and axis for plotting
fig, ax = plt.subplots()
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_aspect('equal')

# Initialize the plot with an empty scatter plot
scat = ax.scatter([], [])


# Function to generate the next set of random dots
def generate_dots() -> object:
    n_dots = 1  # Specify the number of dots
    x = np.random.uniform(0, 100, n_dots)  # Generate random x coordinates
    y = np.random.uniform(0, 100, n_dots)  # Generate random y coordinates
    return x, y


def random_walk(x_pre, y_pre, sigma=30) -> object:
    """
    :param:
    x_pre: 前の集光点
    y_pre: 前の集光点
    :rtype: object
    """
    # 直前の集光点の座標(x0, y0)を受け取って次の集光点(x,y)をランダムに与える
    x_pre += np.random.normal(scale=sigma)  # 平均0，標準偏差: step_sizeの乱数
    y_pre += np.random.normal(scale=sigma)
    x_next = np.clip(x_pre, 0, 100)  # 最小値が0で最大値がx_grid_sizeに指定される
    y_next = np.clip(y_pre, 0, 100)

    return x_next, y_next


def triangle_spot(x_pre, y_pre, triangle_radius=20) -> object:
    """
    閾値を上回った場合に前の集光点を中心に半径self.triangle_radiusの三角形で順に集光する

    :param triangle_radius:
    :param x_pre:
    :param y_pre:

    :return: list (集光点×3のリストであることに注意)
    """
    # 直前の集光点(x0, y0)を中心に半径self.triangle_radiusの三角形で順に集光
    x_list = []
    y_list = []
    # x_list.append(x0)
    # y_list.append(y0)
    for i in range(0, 3):
        x = x_pre + triangle_radius * np.cos(2 * i * np.pi / 3)
        y = y_pre + triangle_radius * np.sin(2 * i * np.pi / 3)
        # print("x= "+"y= ", x, y)
        x_list.append(x)
        y_list.append(y)
    return x_list, y_list

def triangle_spot_seemless(x_pre, y_pre, triangle_radius=5) -> object:
    """
    閾値を上回った場合に前の集光点を中心に半径self.triangle_radiusの三角形で順に集光する

    :param triangle_radius:
    :param x_pre:
    :param y_pre:

    :return: list (集光点×3のリストであることに注意)
    """
    # 直前の集光点(x0, y0)を中心に半径self.triangle_radiusの三角形で順に集光
    x_list = []
    y_list = []
    # x_list.append(x0)
    # y_list.append(y0)
    for i in range(0, 3):
        x = x_pre + triangle_radius * np.cos(2 * i * np.pi / 3)
        y = y_pre + triangle_radius * np.sin(2 * i * np.pi / 3)
        # print("x= "+"y= ", x, y)
        x_list.append(x)
        y_list.append(y)
    max_signal = max()
    return x_list, y_list


# Function to update the scatter plot with the next set of dots
def update():
    x, y = generate_dots()
    scat.set_offsets(np.c_[x, y])
    return scat,


def update_2(i):
    x, y = random_walk(x_pre=50, y_pre=50)
    scat.set_offsets(np.c_[x, y])
    return scat,


x_list, y_list = triangle_spot(x_pre=50, y_pre=50)


def update_4(i):
    scat.set_offsets(np.c_[x_list[i], y_list[i]])
    return scat,


# Create the animation
ani = animation.FuncAnimation(fig, update_2, frames=10, repeat=False)

# Show the animation
plt.show()
