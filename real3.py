import math

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt


def random_walk(x_pre, y_pre, dist='nom'):
    if dist == 'nom':
        x_pre += np.random.normal(scale=30)
        y_pre += np.random.normal(scale=30)
        x_next: ndarray = np.clip(x_pre, 0, 100)
        y_next: ndarray = np.clip(y_pre, 0, 100)
        return x_next, y_next

    if dist == 'uni':
        x_next = np.random.uniform(0, 100)
        y_next = np.random.uniform(0, 100)

        return x_next, y_next


def make_triangle_pos(x_pre_pos, y_pre_pos, pre_intensity=100, triangle_radius=10):
    """
    閾値を上回った座標を中心とした3点の座標

    :param x_pre_pos: 前の集光点のx座標
    :param y_pre_pos: 前の集光点のy座標
    :param pre_intensity: 前の集光点の強度
    :param triangle_radius: 3点計測 (三角形) の外接円の半径

    :return:　x_list, y_list
    """
    x_list = []
    y_list = []
    # 閾値を上回った座標を中心に3点
    for i in range(0, 3):
        x = x_pre_pos + triangle_radius * np.cos(2 * i * np.pi / 3)
        y = y_pre_pos + triangle_radius * np.sin(2 * i * np.pi / 3)
        x_list.append(x)
        y_list.append(y)
    return x_list, y_list


def get_max_index(signal_list):
    max_signal = max(signal_list)    # [sig0, sig1, sig2]の最大値を取得
    max_signal_index = signal_list.index(max_signal)    # 最大の輝度値の時の座標を示すインデックスを取得
    return max_signal_index


# 以下メインパート
x_first = 50    # 視野の中心を想定
y_first = 50    # 視野の中心を想定 (どこでもよい)
random_threshold = 100    # ランダムウォーク時の閾値
keikou_threshold = 150    # 3点集光以降に使う閾値 場合によってはrandom_thresholdと一緒になるかもしれない
count = 0
max_count = 1000
signal = 0
x_next = x_first
y_next = y_first

while signal >= random_threshold:
    # はじめのランダムウォークは適当に座標をわたして輝度値をもらう
    # 閾値判定で以降がランダムウォークか，3点計測か決定する
    x_next, y_next = random_walk(x_pre=x_next, y_pre=y_next)    # 次時点の集光座標を返す
    # 座標をもとに集光，輝度値計測 (米山君)
    # 輝度値(signal)を返してもらう

# 閾値を上回ったときの座標は(x_next, y_next)
# この点を中心にしたいので変数名をx_pre, y_preにそれぞれ変更
x_pre = x_next
y_pre = y_next
x_list, y_list = make_triangle_pos(x_pre_pos=x_pre, y_pre_pos=y_pre)    # 3点の集光点座標を返す

# 座標をもとに3点順に集光，輝度値計測
# 輝度値のリスト signal_list を返してもらう?
while count < max_count:
    count += 1
    # ここで閾値検出後最初の計測 get_signal(x_list, y_list)
    # signalを返してもらう signal_list = [sig0, sig1, sig2]
    # ここで蛍光検出数について条件分岐を挟む
    over_list = [i for i, x in enumerate(signal_list) if x >= keikou_threshold]
    len_over_list = len(over_list)

    if len_over_list == 0:
        # 蛍光検出数が0
        # この時は3点計測の中心≒蛍光ビーズの中心
        # 中心座標は変えずに再度3点計測
        # 3点集光の座標リストだけを返す
        x_list, y_list = make_triangle_pos(x_pre_pos=x_pre, y_pre_pos=y_pre)

    elif len_over_list == 1:
        # 3点計測のうち1点だけ高い輝度値が得られた
        # 3点計測の中心を高い輝度値を検出した方向に移動して3点計測
        # とりあえず前の中心と高輝度値の座標の中点に集光
        over_keikou_threshold_x = x_list[over_list[0]]    # 蛍光の閾値 (keikou_threshold) を上回った時のx座標
        over_keikou_threshold_y = y_list[over_list[0]]    # 蛍光の閾値を上回った時のy座標

        # x_preの値を更新
        # x_preはもともと3点計測の中心点だった
        x_pre = math.ceil((x_pre + over_keikou_threshold_x) / 2)
        y_pre = math.ceil((y_pre + over_keikou_threshold_y) / 2)

        # 3点計測の座標リストを返す
        x_list, y_list = make_triangle_pos(x_pre_pos=x_pre, y_pre_pos=y_pre)

    elif len_over_list == 2:
        # 3点計測のうち2点で高い輝度値が得られた
        # 2点の中間値を次の中心にする
        over_keikou_threshold_x1, over_keikou_threshold_x2 = x_list[over_list[0]], x_list[over_list[1]]
        over_keikou_threshold_y1, over_keikou_threshold_y2 = y_list[over_list[0]], y_list[over_list[1]]

        # x_preの値を更新
        # x_preはもともと3点計測の中心点だった
        # keikou_thresholdを上回った二点の中間座標にx_pre, y_preを更新する
        # ここはmath.ceil((x_pre + over_keikou_threshold_x1 + over_keikou_threshold_x2) / 3)でもいいかも
        x_pre = math.ceil((over_keikou_threshold_x1 + over_keikou_threshold_x2) / 2)
        y_pre = math.ceil((over_keikou_threshold_y1 + over_keikou_threshold_y2) / 2)
        x_list, y_list = make_triangle_pos(x_pre_pos=x_pre, y_pre_pos=y_pre)

    else:
        print("閾値を超えたのが3点 -> triangle_radiusが多分小さすぎる")
        print("いったん終了する")
        break

