import numpy as np
import matplotlib.pyplot as plt

'''

このコードでは、まず2次元空間の大きさと格子点間の距離を定義します。次に、np.meshgrid()を用いてxとyの座標のグリッドを作成します。

次に、np.random.uniform()を用いて、ビーズの中心を表すランダムなx、y座標を生成しています。また、ビーズの直径も指定します。

このコードでは、ビーズの中心からのグリッド上の各点の距離を距離の公式を使って計算しています。
次に、np.exp(-dist**2 / (2 * (bead_diameter/2)**2)) 
という式で、指定した直径のビーズを中心とするガウス関数を作成します。 

最後に、plt.imshow()を用いて、結果の画像をプロットする。cmap='gray' 引数は、画像がグレースケールであることを指定します。



蛍光ビーズから発せられる光の強度分布のモデル化には、一般的にガウス関数が使用される。蛍光ビーズから放射される光の強度は、ビーズの中心からの距離の関数として減少する。
ガウス関数は、この強度分布を表現できる滑らかで連続的な関数である。

蛍光ビーズに光を当てると、光はビーズ内の蛍光分子と相互作用して、より高いエネルギー状態に励起される。この分子が基底状態に戻ると、特定の波長で光を放出する。
発光量はビーズ内の蛍光分子の数に比例し、それはビーズの体積に比例する。したがって、放出される光の強度はビーズの中心部で最も高く、中心部からの距離の関数として減少する。

蛍光ビーズから放射される光の強度分布をガウス関数でモデル化することで、実際の蛍光ビーズの外観をシミュレートし、さまざまな条件下でのビーズの挙動を調べるために利用することができる。
'''

# Define the size of the 2D space and the distance between grid points
x_size = y_size = 100
grid_step = 1

# Create a grid of x and y coordinates
x_grid, y_grid = np.meshgrid(np.arange(0, x_size, grid_step), np.arange(0, y_size, grid_step))

# Generate a random x and y coordinate for the center of the bead
bead_x = np.random.uniform(0, x_size)
bead_y = np.random.uniform(0, y_size)

# Define the diameter of the bead
bead_diameter = 10

# Calculate the distance of each point on the grid from the center of the bead
dist = np.sqrt((x_grid - bead_x)**2 + (y_grid - bead_y)**2)

# Create a Gaussian function centered at the bead with the specified diameter
gaussian = np.exp(-dist**2 / (2 * (bead_diameter/2)**2))

# Plot the resulting image
plt.imshow(gaussian, cmap='gray')
plt.show()
