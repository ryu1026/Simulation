import numpy as np
import matplotlib.pyplot as plt

# 光電子増倍管のパラメータ
photomul_gain = 1e6    # Min:1×10^6, Typ:2.0×10^6
photomul_dark_current = 0
photomul_quantum_efficiency = 0.4    # Min:40%, Typ:45%

# アンプのパラメータ
amp_gain = 200    # 46±2 (約200倍) [dB]
amp_band_width = 10e6
amp_noise_figure = 2

# シミュレーションのパラメータ
sampling_rate = 1e3    # sampling_rate in Hz
time_step = 1 / sampling_rate
signal_duration = 1e2


# 蛍光ビーズからのランダムな信号を生成
t = np.arange(0, signal_duration, time_step)

signal = np.random.normal(loc=1, scale=0.1, size=len(t))

# 信号に光電子増倍管のゲインをかけて，量子効率を乗算する
signal *= photomul_gain * photomul_quantum_efficiency
# 光電子増倍管の暗電流を追加する
signal += photomul_dark_current

# アンプで増幅して周波数帯域を適用
signal *= amp_gain

signal = np.convolve(signal, np.ones(int(amp_band_width * time_step)) / int(amp_band_width * time_step), mode="same")

# アンプのノイズを加える
signal += np.random.normal(scale=10**(amp_noise_figure/20), size=len(signal))
print(len(t))
print(len(signal))
plt.plot(t, signal)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')
plt.show()
