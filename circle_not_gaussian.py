import numpy as np
import matplotlib.pyplot as plt

'''
このコードは、100×100のグリッドの画素を生成し、グリッドのランダムな位置を中心とした直径10ミクロンの円を10個作成します。
次に、各グリッド点から各円の中心までの距離を計算し、各円の内側のピクセルに対してバイナリマスクを作成する。
最後に、すべての円のマスクを合計して、円を含むグレースケールの最終画像を作成する。
得られた画像はmatplotlibを用いて表示される。
'''
# Define the grid
x_grid_size = 100  # microns
y_grid_size = 100  # microns
grid_step = 0.1  # micron
x = np.arange(0, x_grid_size, grid_step)
y = np.arange(0, y_grid_size, grid_step)
X, Y = np.meshgrid(x, y)

# Define the number of circles and their diameter
num_circles = 5
circle_diameter = 10  # microns

# Generate random coordinates for the circles
circle_x = np.random.uniform(0, x_grid_size, num_circles)
circle_y = np.random.uniform(0, y_grid_size, num_circles)

# Calculate the distance from each grid point to the center of each circle
distances = np.sqrt((X[..., np.newaxis] - circle_x)**2 + (Y[..., np.newaxis] - circle_y)**2)

# Create a mask for the pixels inside each circle
mask = distances < (circle_diameter / 2)

# Sum the masks for all the circles to create the final image
image = np.sum(mask, axis=-1)

# Plot the resulting image
plt.imshow(image, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.show()
