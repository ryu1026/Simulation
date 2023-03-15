from simulation import Simulate

sim = Simulate(num_beads=5, grid_step=0.1, spot_diameter=1)

# 蛍光ビーズの空間を作成
beads_matrix = sim.make_not_gaussian_beads(do_print=True)

# 蛍光ビーズの空間を描画する
sim.draw_not_gaussian_beads(not_gaussian_beads=beads_matrix)

# ランダムウォークおよび閾値判定の処理開始
x_first, y_first = 50, 50    # ランダムウォークの初期位置
random_walk_threshold = 1    # ランダムウォーク時に用いる閾値
count = 0    # とりあえずループのカウンタ
max_count = 1000
signal = 0

# 閾値を超えるまでランダムウォーク
while signal >= random_walk_threshold:
    # はじめのランダムウォークはx_first, yを与えて輝度値を取得
    x_next, y_next = sim.random_walk(x_pre=x_next, y_pre=y_next)
    signal =