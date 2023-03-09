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


# 以下メインパート

x_first = 50    # 視野の中心を想定
y_first = 50    # 視野の中心を想定 (どこでもよい)
random_threshold = 100    # ランダムウォーク時の閾値
keikou_threshold = 150    # 3点集光以降に使う閾値 場合によってはrandom_thresholdと一緒になるかもしれない
count = 0
max_count = 100


def get_signal(x_pre, y_pre):
    # はじめのランダムウォークは適当に座標をわたして輝度値をもらう
    # 閾値判定で以降がランダムウォークか，3点計測か決定する
    x_next, y_next = random_walk(x_pre=x_first, y_pre=y_first)  # 次時点の集光座標を返す

    # 座標をもとに集光，輝度値計測 (米山君)
    # 輝度値(signal)を返してもらう
    if signal < random_threshold:
        x_next, y_next = random_walk(x_pre=x_next, y_pre=y_next)

    elif signal >= random_threshold:
        # 閾値を上回ったときの座標は(x_next, y_next)
        # この点を中心にしたいので変数名をx_pre, y_preにそれぞれ変更
        x_pre = x_next
        y_pre = y_next
        x_list, y_list = make_triangle_pos(x_pre_pos=x_pre, y_pre_pos=y_pre)  # 3点の集光点座標を返す

        # 座標をもとに3点順に集光，輝度値計測
        # 輝度値のリスト signal_list を返してもらう?
        max_signal = max(signal_list)  # [sig0, sig1, sig2]の最大値を取得
        max_signal_index = signal_list.index(max_signal)  # 最大の輝度値の時の座標を示すインデックスを取得
        max_x_pos, max_y_pos = x_list[max_signal_index], y_list[max_signal_index]  # 3点の集光点の最大値の座標

        # 3点集光の最大輝度値の座標を中心に3点計測し，輝度値のリストsignal_list2を返してもらう
        x_list2, y_list2 = make_triangle_pos(x_pre_pos=max_x_pos, y_pre_pos=max_y_pos)

        # ここから挙動が変わる
        # 蛍光 (= 閾値以上の輝度値) を検出した計測点を全て使用する
        # 詳しくは以下の条件分岐を参照
        over_list = [i for i, x in enumerate(signal_list2) if x >= keikou_threshold]  # 閾値を上回ったインデクッスをリストで返す
        if len(over_list) == 0:
            # 3点計測いずれからもあまり大きな蛍光が得られなかった
            # この時は3点計測の中心が蛍光ビーズの中心に近いことが推測されるため，中心座標をそのままに3点計測
            x_list, y_list = make_triangle_pos(x_pre_pos=max_x_pos, y_pre_pos=max_y_pos)

        elif len(over_list) == 1:
            # 3点計測のうち一点だけ高い輝度値が得られた
            # 3点計測の中心を高い輝度値を検出した方向に移動して3点計測
            over_keikou_threshold_x = x_list2[over_list[0]]
            over_keikou_threshold_y = y_list2[over_list[0]]

            x_new = math.ceil((max_x_pos + over_keikou_threshold_x) / 2)
            y_new = math.ceil((max_y_pos + over_keikou_threshold_y) / 2)
            x_list, y_list = make_triangle_pos(x_pre_pos=x_new, y_pre_pos=y_new)

        elif len(over_list) == 2:
            # 3点計測のうち2点で高い輝度値が得られた
            # 2点の中間値を次の中心にする
            over_keikou_threshold_x1 = x_list2[over_list[0]]
            over_keikou_threshold_y1 = y_list2[over_list[0]]
            over_keikou_threshold_x2 = x_list2[over_list[1]]
            over_keikou_threshold_y2 = y_list2[over_list[1]]

            x_new = math.ceil((over_keikou_threshold_x1 + over_keikou_threshold_x2) / 2)
            y_new = math.ceil((over_keikou_threshold_y1 + over_keikou_threshold_y2) / 2)
            x_list, y_list = make_triangle_pos(x_pre_pos=x_new, y_pre_pos=y_new)

        else:
            print("閾値を超えたのが3点 -> triangle_radiusが多分デカすぎる")


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
max_signal = max(signal_list)    # [sig0, sig1, sig2]の最大値を取得
max_signal_index = signal_list.index(max_signal)    # 最大の輝度値の時の座標を示すインデックスを取得
max_x_pos, max_y_pos = x_list[max_signal_index], y_list[max_signal_index]    # 3点の集光点の最大値の座標

# 3点集光の最大輝度値の座標を中心に3点計測し，輝度値のリストsignal_list2を返してもらう
x_list2, y_list2 = make_triangle_pos(x_pre_pos=max_x_pos, y_pre_pos=max_y_pos)

# ここから挙動が変わる
# 蛍光 (= 閾値以上の輝度値) を検出した計測点を全て使用する
# 詳しくは以下の条件分岐を参照
over_list = [i for i, x in enumerate(signal_list2) if x >= keikou_threshold]    # 閾値を上回ったインデクッスをリストで返す
if len(over_list) == 0:
    # 3点計測いずれからもあまり大きな蛍光が得られなかった
    # この時は3点計測の中心が蛍光ビーズの中心に近いことが推測されるため，中心座標をそのままに3点計測
    x_list, y_list = make_triangle_pos(x_pre_pos=max_x_pos, y_pre_pos=max_y_pos)

elif len(over_list) == 1:
    # 3点計測のうち一点だけ高い輝度値が得られた
    # 3点計測の中心を高い輝度値を検出した方向に移動して3点計測
    over_keikou_threshold_x = x_list2[over_list[0]]
    over_keikou_threshold_y = y_list2[over_list[0]]

    x_new = math.ceil((max_x_pos + over_keikou_threshold_x) / 2)
    y_new = math.ceil((max_y_pos + over_keikou_threshold_y) / 2)
    x_list, y_list = make_triangle_pos(x_pre_pos=x_new, y_pre_pos=y_new)

elif len(over_list) == 2:
    # 3点計測のうち2点で高い輝度値が得られた
    # 2点の中間値を次の中心にする
    over_keikou_threshold_x1 = x_list2[over_list[0]]
    over_keikou_threshold_y1 = y_list2[over_list[0]]
    over_keikou_threshold_x2 = x_list2[over_list[1]]
    over_keikou_threshold_y2 = y_list2[over_list[1]]

    x_new = math.ceil((over_keikou_threshold_x1 + over_keikou_threshold_x2) / 2)
    y_new = math.ceil((over_keikou_threshold_y1 + over_keikou_threshold_y2) / 2)
    x_list, y_list = make_triangle_pos(x_pre_pos=x_new, y_pre_pos=y_new)

else:
    print("閾値を超えたのが3点 -> triangle_radiusが多分デカすぎる")
