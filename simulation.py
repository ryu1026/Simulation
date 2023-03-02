import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import random


class Simulate:
    def __init__(self,
                 num_beads=1,
                 beads_diameter=10,
                 spot_diameter=1,
                 threshold=0.5,
                 max_iteration=100,
                 x_grid_size=100,
                 y_grid_size=100,
                 grid_step=0.5,
                 triangle_radius=5,
                 walk_step_size=1,
                 intensity_max=1):

        self.num_beads = num_beads
        self.beads_diameter = beads_diameter
        self.spot_diameter = spot_diameter
        self.threshold = threshold
        self.max_iteration = max_iteration
        self.x_grid_size = x_grid_size
        self.y_grid_size = y_grid_size
        self.grid_step = grid_step
        self.walk_step_size = walk_step_size
        self.triangle_radius = triangle_radius
        self.x0 = self.x_grid_size // 2
        self.y0 = self.y_grid_size // 2
        self.intensity_max = intensity_max

    def make_grid_and_gaussian_beads(self):
        """
        グリッド (self.x_grid_size×self.y_grid_size) とガウス関数で近似した蛍光ビーズを作成する
        蛍光ビーズはnp.clip(beads, 0, 1)で0から1に範囲指定を行っていることに注意

        Return:
        -------
        numpy.ndarray
            グリッド上に配置された蛍光ビーズ
        """

        # gird_xは列方向に同じ値のベクトルを作成 ->
        # [[0, 0.1, 0.2]
        #  [0, 0.1, 0.2]]
        # gird_yは行方向に同じ値のベクトルを作成
        # [[0,     0,   0]
        #  [0.1, 0.1, 0.1]]
        # 作成された格子点 左上が原点のx-y二次元空間のイメージ
        # xは右に行くほど，yは下に行くほど値が増大する
        # (0,   0) (0.1,   0) (0.2,   0)
        # (0, 0.1) (0.1, 0.1) (0.2, 0.1)
        # (0. 0.2) (0.1, 0.2) (0.2, 0.2)
        grid_x, grid_y = np.meshgrid(np.arange(0, self.x_grid_size, self.grid_step),
                                     np.arange(0, self.y_grid_size, self.grid_step))

        # 0~grid_sizeの範囲でビーズの個数分=num_beads個の乱数を出す
        # 蛍光ビーズの座標をランダムに生成
        beads_x = np.random.uniform(0, self.x_grid_size, self.num_beads)
        beads_y = np.random.uniform(0, self.y_grid_size, self.num_beads)

        beads = np.zeros_like(grid_x)
        clip_beads = np.zeros_like(grid_x)
        # 各ビーズについて，ビーズの座標を中心とした2次元ガウス関数で強度を返す
        # ガウス関数は，グリッド上の各点におけるビーズの蛍光強度を示す．
        for bead_x, bead_y in zip(beads_x, beads_y):

            dist = np.sqrt((grid_x - bead_x)**2 + (grid_y - bead_y)**2)
            bead_z = np.exp(-dist ** 2 / (2 * (self.beads_diameter/2) ** 2))
            # bead_z = np.exp(-((grid_x - bead_x)**2 + (grid_y - bead_y)**2) / (2 * (self.beads_diameter/2)**2))
            beads += bead_z
            clip_beads = np.clip(beads, 0, 1)

        return beads, clip_beads

    def make_grid_and_not_gaussian_beads(self, bead_central_position=False):
        """
        グリッドと蛍光ビーズを作成する
        蛍光ビーズ内の蛍光体の分布は一様であると仮定したバージョン
        """
        grid_x, grid_y = np.meshgrid(np.arange(0, self.x_grid_size, self.grid_step),
                                     np.arange(0, self.y_grid_size, self.grid_step))

        beads_x = np.random.uniform(0, self.x_grid_size, self.num_beads)
        beads_y = np.random.uniform(0, self.y_grid_size, self.num_beads)

        mask = np.zeros_like(grid_x)
        clip_mask = np.zeros_like(grid_x)

        for bead_x, bead_y in zip(beads_x, beads_y):

            # 中心座標を表示
            # 作成した格子はy座標は下に行くほど値が大きいが
            # 画像では原点が左下にありy座標は上に行くほど値が大きくなるからabs(y_grid-size_y)で調整
            if bead_central_position:
                print("x:" + str(bead_x) + "  y:" + str(abs(self.y_grid_size-bead_y)))

            # ランダムに決定した中心と格子点の距離を計算
            distances = np.sqrt((grid_x - bead_x)**2 + (grid_y - bead_y)**2)
            mask += distances < (self.beads_diameter / 2)
            # plt.imshow(mask, cmap='gray')
            # plt.show()
            # image = np.sum(mask, axis=-1)
            clip_mask = np.clip(mask, 0., 1.)
        return grid_x, grid_y, mask, clip_mask

    def draw_not_gaussian_beads(self, not_gaussian_beads):
        plt.imshow(not_gaussian_beads, cmap='gray', interpolation='nearest', extent=(0, self.x_grid_size, 0, self.y_grid_size))
        plt.xlabel("x (microns)")
        plt.ylabel("y (microns)")
        plt.title("Not gaussian fluorescent beads")
        plt.show()

    def draw_fluo_beads(self, beads):
        fig, ax = plt.subplots()
        ax.imshow(beads, cmap='gray', extent=(0, self.x_grid_size, 0, self.y_grid_size))
        ax.set_xlabel('x (microns)')
        ax.set_ylabel('y (microns)')
        ax.set_title('Gaussian fluorescent beads')
        plt.show()

    def random_walk(self, x_pre, y_pre):
        # 直前の集光点の座標(x0, y0)を受け取って次の集光点(x,y)をランダムに与える
        x_pre += np.random.normal(scale=self.walk_step_size)  # 平均0，標準偏差: step_sizeの乱数
        y_pre += np.random.normal(scale=self.walk_step_size)
        x_next = np.clip(x_pre, 0, self.x_grid_size)  # 最小値が0で最大値がx_grid_sizeに指定される
        y_next = np.clip(y_pre, 0, self.y_grid_size)
        return x_next, y_next

    def triangle_spot(self, x_pre, y_pre):
        # 直前の集光点(x0, y0)を中心に半径self.triangle_radiusの三角形で順に集光
        x_list = []
        y_list = []
        # x_list.append(x0)
        # y_list.append(y0)
        for i in range(0, 3):
            x = x_pre + self.triangle_radius * np.cos(2*i*np.pi/3)
            y = y_pre + self.triangle_radius * np.sin(2*i*np.pi/3)
            # print("x= "+"y= ", x, y)
            x_list.append(x)
            y_list.append(y)
        return x_list, y_list

    def generate_spot(self, grid_x, grid_y, x, y, draw=False):
        """
        詳細はbeam_sim.pyを参照
        集光点の中心座標をもとにガウシアンビームを作成する

        :param grid_x: ビーズをつくったときにできた格子点
        :param grid_y: ビーズを作ったときにできた格子点
        :param x: 集光点のx座標
        :param y: 集光点のy座標
        :param draw: bool 作成したスポットを描画する

        :return: 中心(x,y)でスポットの直径がself.spot_diameterのビーム（強度）
        """

        # sigma = self.spot_diameter / (2 * np.sqrt((2*np.log(2)))) / self.grid_step
        sigma = self.spot_diameter / (2 * np.sqrt((2*np.log(2))))
        # グリッドと実際のy座標は増大方向が反対だから差の絶対値をとる
        y = abs(self.y_grid_size - y)

        r = np.sqrt((grid_x - x)**2 + (grid_y - y)**2)

        spot_intensity = self.intensity_max * np.exp(-r**2 / (2 * sigma**2))
        # spot = np.exp(-((x-self.x0)**2 + (y-self.y0)**2) / (2 * (self.spot_diameter/2)**2))

        # 作成したスポットを確認するかどうか
        if draw:
            plt.imshow(spot_intensity, cmap='hot', extent=(0, self.x_grid_size, 0, self.y_grid_size))
            # plt.imshow(spot_intensity, cmap='hot', extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])
            plt.xlabel('x (µm)')
            plt.ylabel('y (µm)')
            plt.colorbar()
            plt.show()

        return spot_intensity

    def draw_beads_and_spot_animation(self, beads, spot_intensity):

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_xlabel('x [µm]')
        ax.set_ylabel('y [µm]')
        ax.set_title('Spot animation')

        imgs = []
        ax.imshow(beads, cmap='gray', extent=(0, self.x_grid_size, 0, self.y_grid_size))
        img2 = beads + spot_intensity
        # plt.imshow(img, cmap='gray', extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])
        # imgs.append([img])
        img2 = ax.imshow(img2, cmap='hot', extent=(0, self.x_grid_size, 0, self.y_grid_size))
        imgs.append([img2])

        ani = animation.ArtistAnimation(fig, imgs)
        plt.show()

    def get_signal(self, spot):
        """
        :param spot:
        :return: 実際の
        """
        return signal

    def get_signal_and_judge_threshold(self, x, y):
        # 光を(x,y)に当てて蛍光をPMTで収集→Ampで増幅
        """

        :param x:
        :param y:
        :return:
        """
        count = 0
        while True:
            # まず視野の中心でスポットを作って信号を取得する
            # スポットはgenerate_spotメソッドでつくる
            # 信号取得はget_signalメソッドでつくる
            count += 1
            spot = np.exp(-((x-self.x0)**2 + (y-self.y0)**2) / (2 * (self.spot_diameter/2)**2))
            signal = np.sum(z * spot)

            # 閾値判定でランダムウォークを続けるか決める
            if signal >= self.threshold:
                # triangle_spotメソッドからは3点のシーケンスがそれぞれリストとして渡される
                # next_spot_x，next_spot_yはそれぞれ長さが3のリスト
                next_spot_x, next_spot_y = self.triangle_spot(x, y)
                for i in range(0, 3):
                    self.get_signal_and_judge_threshold(int(next_spot_x[i]), int(next_spot_y[i]))
            else:
                self.random_walk(x, y)

            if count == self.max_iteration:
                print("Done max iteration!!")
                break

    def after_over_threshold_get_signal(self, next_spot_x, next_spot_y):
        # 閾値を上回った場合の三角形照射における信号取得メソッド
        # 周囲の3点の内, 最大値の信号を持つ集光点の座標を返す
        max_x = next_spot_x    # 最大値の信号を持つ集光点のx座標
        max_y = next_spot_y    # 最大値の信号を持つ集光点のy座標
        max_signal = self.threshold       # 暫定の最大値信号

        for i in range(0, len(next_spot_x)):
            # 中心(x,y)にガウシアンビーム生成
            # 本来はgenerate_spotメソッドでスポット作成
            spot = np.exp(-((x-next_spot_x[i])**2 + (y-next_spot_y[i])**2) / (2 * (self.spot_diameter/2)**2))
            # spot = generate_spot()
            # 本来はget_signalメソッドで信号取得
            signal = np.sum(z * spot)
            # signal = get_signal()

            # 信号の最大値が更新されたら, その時の集光点座標を記録
            # 最初はmax_signal=self.thresholdで定義してある
            if signal > max_signal:
                max_x = x
                max_y = y
                # 信号の最大値を変更
                max_signal = signal

        # 信号の最大値が更新されたら中心を(max_x, max_y)に変更して同じ作業を繰り返す
        if max_signal != self.threshold:
            self.after_over_threshold_get_signal(max_x, max_y)
        # 信号の最大値が更新されなかった = 誤検出?
        # あるいは探索範囲が広すぎた? -> self.radiusを縮めて探索範囲を限定する
        else:
            self.triangle_radius -= 2
            self.triangle_spot()
