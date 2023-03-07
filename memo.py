

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