import numpy as np
import matplotlib.pyplot as plt

#このコードは、n_beadsのランダムな位置とビーズの蛍光強度を生成し、
# 与えられた波長とパワーでガウス型励起プロファイルを計算します。
# この励起プロファイルを用いて、各ビーズからの蛍光シグナルを計算し、ガウスノイズを付加します。
# 蛍光信号は合計されて全体の蛍光信号となり、これを発光フィルターで畳み込んで最終的な発光信号とする。
# 励起プロファイル、蛍光シグナル、発光シグナルは、可視化のためにプロットされる。
# これは簡略化した例であり、光退色、光毒性、特定の蛍光色素の特性などの要素を考慮していないことに注意してください。
# しかし、蛍光ビーズの励起と発光をシミュレートするための出発点となるものである。

import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_beads = 2
bead_pos = np.random.rand(n_beads, 2) * 100
excitation_wavelength = 488  # in nm
emission_wavelength = 525  # in nm
quantum_yield = 0.9
NA = 1.4
pixel_size = 100 / 100  # in microns
emission_filter_width = 20  # in nm

# Generate grid for calculating signal
x, y = np.meshgrid(np.linspace(0, 100, 101), np.linspace(0, 100, 101))
bead_pos = np.transpose(bead_pos)  # transpose to shape (2, 1000)
r = np.sqrt((x[:, :, np.newaxis] - bead_pos[:, 0])**2 + (y[:, :, np.newaxis] - bead_pos[:, 1])**2)


# Calculate excitation intensity
excitation_intensity = np.exp(-(2 * np.pi * NA * r / excitation_wavelength)**2 / 2)

# Calculate fluorescence signal
emission_intensity = quantum_yield * excitation_intensity / \
                     (1 + 2 * (np.pi**2) * (emission_filter_width / 2.355)**2 * (excitation_wavelength
                                                                                 / emission_wavelength)**4)
total_signal = np.sum(emission_intensity, axis=2) * pixel_size**2

# Plot signal
fig, ax = plt.subplots()
im = ax.imshow(total_signal, cmap='hot')
ax.set_title('Fluorescent Beads Signal')
fig.colorbar(im, ax=ax, label='Signal (photons)')
plt.show()

