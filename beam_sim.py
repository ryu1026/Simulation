import numpy as np
import matplotlib.pyplot as plt

# Define parameters
diameter = 10  # µm
center = (0, 0)  # µm
pixel_size = 0.1  # µm
intensity_max = 1.0  # maximum intensity

# この計算の目的は、光度分布を生成するために使用するガウス関数の標準偏差を求めることです。
# シグマ値は、ガウス関数の広がりを表し、光スポットの幅を決定する。
# この計算では、光スポットの直径（マイクロメートル）を、一定値である2の自然対数の平方根の2倍で割る。
# そして、得られた値を画素サイズで割って、直径をグリッド上でスポットを表現するために必要な画素数に変換する。
# そして、そのシグマ値をガウス関数で使用する。
sigma = diameter / (2 * np.sqrt(2 * np.log(2))) / pixel_size  # standard deviation

# Generate coordinates
x = np.arange(-50, 50, pixel_size)
y = np.arange(-50, 50, pixel_size)
xx, yy = np.meshgrid(x, y)
r = np.sqrt((xx - center[0])**2 + (yy - center[1])**2)

# ここでは、スポットの中心を基準とした格子上の各点での光の強度をガウス関数で求めています。
# 各点の強度は、スポットの最大強度（intensity_max）に、スポットの中心からの距離（r）の指数関数を掛け、
# 標準偏差（σ）の二乗で割った値で計算する。スポットの中心に近い点ほど、その強度値は高くなる。
# 全体として、シグマ値は光スポットの幅を決定するため、シミュレーションの解像度と精度に影響を及ぼします
# シグマ値が小さいほど、ガウス関数が狭くなり、光強度分布のピークが鋭くなるため、
# 光スポットの焦点を絞ったシミュレーションをより正確に行うことができます。
# Calculate intensity distribution
intensity = intensity_max * np.exp(-r**2 / (2 * sigma**2))

# Plot results
plt.imshow(intensity, cmap='gray', extent=[x.min()-200, x.max()+200, y.min()-200, y.max()+200])
plt.xlabel('x (µm)')
plt.ylabel('y (µm)')
plt.colorbar()
plt.show()
