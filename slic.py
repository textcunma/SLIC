# SLIC再現実装
import math
import tqdm
import argparse
import numpy as np
from copy import deepcopy

# utils.pyに実装した関数たち
from utils import (
    loadImg,
    displayCenter,
    displayCluster,
    displayClusterContour,
    displayClusterColor,
    bgr2labImg,
)


class SLIC:
    def __init__(self, k, max_itr, th, img):
        """
        :param k (type:int) クラスタ数
        :param max_itr (type:int) イテレーション最大数
        :param img (type:numpy.ndarray) Lab画像
        """
        # クラスタ数、イテレーション最大数
        self.k = k      # k -> k-meansの"k"
        self.max_itr = max_itr  # itr -> iterationの略

        # 色情報と位置情報のバランスを調整
        self.m = 10

        # 画像の高さと幅、ピクセル数
        self.h = img.shape[0]
        self.w = img.shape[1]
        self.N = self.h * self.w    # 論文表記に合わせる

        # グリッドの間隔
        self.S = int(math.sqrt(self.N / self.k))    # 論文表記に合わせる

        # ピクセル情報:[l,a,b,x,y]の形式
        self.pixels = self.createPixelInfo(img)

        # 各ピクセルがどのクラスタに属しているかをラベリング
        self.pixels_label = np.full((self.h, self.w), None)

        # 各ピクセルとクラスタ重心との距離
        self.pixels_distance = np.full((self.h, self.w), math.inf)

        # 重心移動量の閾値とフラグ
        self.th = th    # th -> thresholdの略称
        self.th_flg = False

        # クラスタの重心座標
        self.cluster_center = np.array([], dtype=int)
        self.calcInitCenter()

        # クラスタ数の変更
        self.k = len(self.cluster_center)
        print("スーパーピクセル数:", self.k)
        displayCenter(self.cluster_center, deepcopy(img))  # 重心座標を描画

        # # 勾配を基に重心座標を調整
        self.transformCenter()
        displayCenter(self.cluster_center, deepcopy(img))  # 重心座標を描画

    def __call__(self):
        """
        メイン実行
        :return cluster_center (type:numpy.ndarray, int32) 各クラスタ重心のlabxy情報
        :return pixels_label (type:numpy.ndarray, Object) 各画素の属しているクラスタの情報
        """
        # 最大イテレーション数まで繰り返し実行
        for i in tqdm.tqdm(range(self.max_itr)):
            if self.th_flg:
                print("重心移動変化量が閾値以下になったため処理終了")
                break
            self.calcNewCenter()
            print(i)

        return self.cluster_center, self.pixels_label

    def createPixelInfo(self, img):
        """
        [l,a,b]から[l,a,b,x,y]の形式に変更
        :param img (type:numpy.ndarray, uint8) Lab画像
        :return [l,a,b,x,y]形式にしたピクセル情報 shape=(H,W,5)
        """
        # ※画素の位置は[y][x]の順に指定する必要があることに注意

        return np.array(
            [
                [img[y][x][0], img[y][x][1], img[y][x][2], x, y]
                for y in range(self.h)
                for x in range(self.w)
            ]
        ).reshape([self.h, self.w, 5])

    def calcInitCenter(self):
        """
        クラスタの重心位置の初期化計算
        """
        # グリッドの行数、列数を求める
        k_w = math.ceil(math.sqrt(self.k * self.w / self.h))
        k_h = math.ceil(math.sqrt(self.k * self.h / self.w))

        # クラスタ数を更新
        self.k = k_w * k_h
        self.S = int(math.sqrt(self.N / self.k))

        # 等間隔で重心を配置
        offset_x = int((self.w - self.S * (k_w - 1)) / 2)
        offset_y = int((self.h - self.S * (k_h - 1)) / 2)
        for h in range(k_h):
            for w in range(k_w):
                y = h * self.S + offset_y
                x = w * self.S + offset_x
                self.cluster_center = np.append(self.cluster_center, self.pixels[y][x])
        self.cluster_center = self.cluster_center.reshape([-1, 5])

    def transformCenter(self):
        """
        勾配を基にクラスタの重心を移動
        """

        def I(x, y):
            return self.pixels[y][x][:3]  # Labベクトルを抽出

        for i in range(len(self.cluster_center)):
            # 重心座標取得
            x, y = self.cluster_center[i][-2:]
            # 基準となる勾配を初期化
            base_grad = float("inf")
            # 重心座標の基準を決定
            base_x, base_y = x, y

            # 移動変化量
            dscope = [
                (0, 0),
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (-1, -1),
                (1, -1),
                (-1, 1),
            ]

            # 3×3マスの勾配を全て計算、最小な勾配の位置を重心にする
            for d in dscope:
                cur_x, cur_y = x + d[0], y + d[1]
                # 勾配を計算
                vect1 = I(cur_x + 1, cur_y) - I(cur_x - 1, cur_y)
                vect2 = I(cur_x, cur_y + 1) - I(cur_x, cur_y - 1)
                g = (
                    np.linalg.norm(vect1, ord=2) ** 2
                    + np.linalg.norm(vect2, ord=2) ** 2
                )
                # 最小な勾配を選択して更新
                if base_grad > g:
                    base_grad = g
                    base_x = cur_x
                    base_y = cur_y

            # 最小な勾配であった位置を重心にする
            l, a, b = self.pixels[base_y][base_x][:3]
            self.cluster_center[i] = [l, a, b, base_x, base_y]

    def calcDistance(self, center, point):
        """
        スーパーピクセルのサイズを考慮した距離計算
        :param center 重心情報[l,a,b,x,y]
        :param point 画素情報[l,a,b,x,y]
        :return 色と位置に関する距離 (type:numpy.float64)
        """
        # 重心のlab値・位置取得
        center_l, center_a, center_b, center_x, center_y = center
        # 点のlab値・位置を取得
        l, a, b, x, y = point

        # 距離計算
        d_lab = np.sqrt((center_l - l) ** 2 + (center_a - a) ** 2 + (center_b - b) ** 2)
        d_xy = np.sqrt((center_x - x) ** 2 + (center_y - y) ** 2)
        return d_lab + (self.m / self.S) * d_xy

    def calcNewCenter(self):
        """
        局所クラスタリング -> 新たな重心位置の決定と移動
        """
        # 局所クラスタリング
        for center_idx, center in enumerate(self.cluster_center):
            # クラスタ重心の位置を取得
            (center_x, center_y) = center[-2:]
            center_x = int(center_x)
            center_y = int(center_y)

            # 画像幅・高さを超えないように範囲を制限
            x_lower = max(0, center_x - self.S)
            x_upper = min(self.w, center_x + self.S)
            y_lower = max(0, center_y - self.S)
            y_upper = min(self.h, center_y + self.S)

            # 各重心に近い点をそのクラスタに属するとする
            for y in range(y_lower, y_upper):  # y-S <= y <= y+S
                for x in range(x_lower, x_upper):  # x-S <= x <= x+S
                    d = self.calcDistance(self.pixels[y][x], center)
                    if d < self.pixels_distance[y][x]:
                        self.pixels_distance[y][x] = d
                        self.pixels_label[y][x] = center_idx

        move_value = []  # 全クラスタの重心が1ステップでどれだけ移動したかを記録

        for l in range(self.k):
            idxs = np.where(self.pixels_label == l)  # 各クラスタに属するピクセル座標[y,x]
            cnt = len(idxs[0])  # 各クラスタに属するピクセルの個数
            avg_y = np.round(np.sum(idxs[0]) / cnt)
            avg_x = np.round(np.sum(idxs[1]) / cnt)
            idxs = np.stack([idxs[0], idxs[1]], 1)  # x座標とy座標の情報を結合
            avg_l = np.round(np.sum(self.pixels[tuple((idxs).T)][:, 0]) / cnt)
            avg_b = np.round(np.sum(self.pixels[tuple((idxs).T)][:, 2]) / cnt)
            avg_a = np.round(np.sum(self.pixels[tuple((idxs).T)][:, 1]) / cnt)

            next_center = np.array([avg_l, avg_a, avg_b, avg_x, avg_y], dtype=int)
            # 重心移動量(マンハッタン距離)を計算
            move_value.append(
                np.linalg.norm(self.cluster_center[l] - next_center, ord=1)
            )
            # 重心移動
            self.cluster_center[l] = next_center

        # 重心移動量(マンハッタン距離)が閾値以下になれば処理終了
        if self.th > min(move_value):
            self.th_flg = True


def main(args):
    # 画像読み込み
    bgrimg = loadImg(mode=args.mode, img_url=args.img_url, img_path=args.img_path)
    # BGR画像をLab画像に変換
    labimg = bgr2labImg(bgrimg)

    # SLICアルゴリズム実行
    slic = SLIC(k=args.k, max_itr=args.max_itr, th=args.th, img=labimg)
    center, label = slic()
    displayCenter(center, deepcopy(bgrimg))  # 重心座標を描画
    displayCluster(center, label, deepcopy(labimg))  # クラスタを画像内の色で描画
    displayClusterColor(center, label, deepcopy(labimg))  # クラスタをランダム色で描画
    displayClusterContour(label, deepcopy(bgrimg))  # クラスタの輪郭を描画


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLIC")
    parser.add_argument("--k", type=int, default=50, help="クラスタ数")
    parser.add_argument("--th", type=float, default=0.001, help="処理終了閾値")
    parser.add_argument("--max_itr", type=int, default=10, help="イテレーション最大数")
    parser.add_argument("--save", action="store_true", help="画像保存フラグ")
    parser.add_argument(
        "--img_path", type=str, default="test.jpg", help="ローカル上にある画像ファイルパス"
    )
    parser.add_argument(
        "--img_url",
        type=str,
        default="http://www.lenna.org/len_std.jpg",
        help="インターネット上にある画像URL",
    )
    parser.add_argument(
        "--mode", type=str, choices=["net", "local"], default="net", help="入力画像先"
    )
    args = parser.parse_args()
    main(args)
