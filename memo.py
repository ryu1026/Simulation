

signal_list = []
for i in range(0, 3):
    spot = self.generate_spot(next_spot_x[i], next_spot_y[i])
    signal = self.get_signal(signal, ... ,)
max_signal = max(signal)
max_signal_index = signal.index(max_signal)
next_center_spot_x, next_center_spot_y = next_spot_x[max_signal_index], next_spot_y[max_signal_index]
x_list, y_list = self.triangle_spot(next_center_spot_x, next_center_spot_y)



# 閾値を超えたときの処理 -> 実質追跡か
# 閾値を超えた時の集光点を中心にトライアングルスポットを作成
# スポットの座標はtriangle_spot(x_pre, y_pre)で算出
# x_list, y_listがgenerate_spotから返ってくる
# generate_spotメソッドにx_list,y_listから座標を取得しspotに代入
# get_signalで光電子増倍管，アンプを通して輝度値を取得
# signal.index(max(signal))で3点の計測点の内最大値を示したインデックスを返す
# continueでつづけることもできえる
# 一方、ある条件で処理をさせて、ストップさせずにそのまま処理させたい（while文を続けたい）ときには、continueを使います。
# 最大値を示したインデックスを中心にtriangle_spotつくる
# 終
def after_over_threshold_get_signal(self, over_threshold_x, over_threshold_y):
    """
    :param self:
    :param over_threshold_x: 閾値を超えたときの集光点x座標
    :param over_threshold_y: 閾値を超えたときの集光点のy座標
    :return:
    """
    x_next_list, y_next_list = self.triangle_spot(over_threshold_x, over_threshold_y)
    signal = []
    for i in range(len(x_next_list)):
        spot = self.generate_spot(x_next_list[i], y_next_list[i])
        signal = get_signal(spot)
    max_signal_index = signal.index(max(signal))
    after_over_threshold_get_signal(x_next_list[max_signal_index], y_next_list[max_signal_index])

def get_signal_general(x_list, y_list):
    # 3点計測の結果がsignal_listに格納される
    # ここでは蛍光検出数が0か1か2の3通りに分類される
    signal_list = get_signal(x_list, y_list)
    max_signal_index = get_max_index(signal_list=signal_list)
    max_x_pos, max_y_pos = x_list[max_signal_index], y_list[max_signal_index]
    x_list2, y_list2 = make_triangle_pos(x_pre_pos=max_x_pos, y_pre_pos=max_y_pos)
    signal_list2 = get_signal(x_lis2, y_list2)
    return max_x_pos, max_y_pos, signal_list2


def get_signal_general2(x_list, y_list, keikou_threshold=10):
    # 3点計測の結果がsignal_listに格納される
    # ここでは蛍光検出数が0か1か2の3通りに分類される
    signal_list = get_signal(x_list, y_list)
    over_list = [i for i, x in enumerate(signal_list) if x >= keikou_threshold]
    if len(over_list==0):
        x_list, y_list = make_triangle_pos(x_pre_pos=max_x_pos, y_pre_pos=max_y_pos)
        signal_list = get_signal_not_keikou_thredshold(x_list, y_list)
    max_signal_index = get_max_index(signal_list=signal_list)
    max_x_pos, max_y_pos = x_list[max_signal_index], y_list[max_signal_index]
    x_list2, y_list2 = make_triangle_pos(x_pre_pos=max_x_pos, y_pre_pos=max_y_pos)
    signal_list2 = get_signal(x_lis2, y_list2)
    return max_x_pos, max_y_pos, signal_list2


def get_signal_not_keikou_thredshold(x_list, y_list):
    # 蛍光検出数が0の時のシグナルのとり方
    signal_list = get_signal(x_list, y_list)
    return signal_list

count = 0
max_count = 100

while count < max_count:
    count += 1
    if count == 1:
        max_x_pos, max_y_pos, signal_list = get_signal_general(x_list, y_list)

    # signal_listの内，keikou_thresholdを上回った要素のインデックスが格納されている
    over_list = [i for i, x in enumerate(signal_list) if x >= keikou_threshold]

    if len(over_list) == 0:
        x_list, y_list = make_triangle_pos(x_pre_pos=max_x_pos, y_pre_pos=max_y_pos)
        signal_list = get_signal_not_keikou_thredshold(x_list, y_list)

    elif len(over_list) == 1:
        over_keikou_threshold_x = x_list2[over_list[0]]
        over_keikou_threshold_y = y_list2[over_list[0]]

        x_new = math.ceil((max_x_pos + over_keikou_threshold_x) / 2)
        y_new = math.ceil((max_y_pos + over_keikou_threshold_y) / 2)
        # 中心は前の中心と蛍光を観測した場所との中間
        x_list, y_list = make_triangle_pos(x_pre_pos=x_new, y_pre_pos=y_new)
        max_x_pos, max_y_pos, signal_list = get_signal_general(x_list, y_list)

    elif len(over_list) == 2:
        over_keikou_threshold_x1 = x_list2[over_list[0]]
        over_keikou_threshold_y1 = y_list2[over_list[0]]
        over_keikou_threshold_x2 = x_list2[over_list[1]]
        over_keikou_threshold_y2 = y_list2[over_list[1]]

        x_new = math.ceil((over_keikou_threshold_x1 + over_keikou_threshold_x2) / 2)
        y_new = math.ceil((over_keikou_threshold_y1 + over_keikou_threshold_y2) / 2)

        x_list, y_list = make_triangle_pos(x_pre_pos=x_new, y_pre_pos=y_new)
        max_x_pos, max_y_pos, signal_list = get_signal_general(x_list, y_list)

