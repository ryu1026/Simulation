from simulation import Simulate
import matplotlib.pyplot as plt
import numpy as np
# ガウシアンでないかつ動いていない蛍光ビーズを適切に探索できるかをチェックする
# はじめに蛍光ビーズの空間を定義している
sim = Simulate(num_beads=1, grid_step=0.1, spot_diameter=1)

# ここで返ってくるbeads_col, beads_rowはグリッドの細かさを考慮していない
# またbeads_colはnumpy.ndarrayで返ってくる事に注意
# グラフにした時のイメージ (100×100 µmだけど1000×1000が配列になっている)
# よって，beads_colの値をgridに合わせるためには，beads_col / self.grid_stepをしないといけない
beads_matrix, beads_col, beads_row = sim.make_not_gaussian_beads(do_print=True)

beads_matrix_is_1 = np.where(beads_matrix == 1)

# print("beads_matrix_is_1", beads_matrix_is_1)
# print("beads_matrix_is_1[0].min:{0},max:{1},mean:{2}".format(beads_matrix_is_1[0].min(), beads_matrix_is_1[0].max(),
#                                                              beads_matrix_is_1[0].mean()))
# print("beads_matrix_is_1[1].min:{0},max:{1},mean:{2}".format(beads_matrix_is_1[1].min(), beads_matrix_is_1[1].max(),
#                                                              beads_matrix_is_1[1].mean()))
# sim.draw_not_gaussian_beads(not_gaussian_beads=beads_matrix)
signal = sim.get_signal_simple(beads_matrix, col_pos=int(beads_col), row_pos=int(beads_row))
print(signal)

