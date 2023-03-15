from simulation import Simulate
import matplotlib.pyplot as plt
import numpy as np
# ガウシアンでないかつ動いていない蛍光ビーズを適切に探索できるかをチェックする
# はじめに蛍光ビーズの空間を定義している

sim = Simulate(num_beads=1, grid_step=0.1, spot_diameter=1)
beads_matrix= sim.make_not_gaussian_beads(do_print=True)
signal = sim.get_signal(beads_matrix, 10, 20)
print(type(signal))
print("signal.shape[0]:", signal.shape[0])
print("signal.shape[1]:", signal.shape[1])
print((np.arange(signal.shape[0])-10)**2)
print((np.arange(signal.shape[1])-20)**2)

"""
t = np.arange(beads_matrix.shape[0])
print(t.shape)
# sim.draw_not_gaussian_beads(clipped_mask)
spot_intensity, spot_pos = sim.generate_spot(10, 10, do_threshold=True, do_draw=False)
print(type(spot_pos))
print(spot_pos)
"""
# sim.draw_beads_and_spot_animation_2(clipped_mask, spot_intensity)
'''
# 蛍光ビーズの空間を取得
not_gaussian_grid_x, not_gaussian_grid_y, not_gaussian_beads, not_gaussian_clip_beads \
    = sim.make_grid_and_not_gaussian_beads()

# 蛍光ビーズ空間を画像で表示
sim.draw_not_gaussian_beads(not_gaussian_beads=not_gaussian_clip_beads)
'''