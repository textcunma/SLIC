import cv2
import numpy as np
import urllib.request
import multiprocessing as mp


def loadImg(mode, img_url, img_path):
    """
    画像を読み込む
    :param mode (type:string) "net" or "local" ネット上にある画像orローカルファイル を読み込み
    :param img_url (type:string) ネットの画像url
    :param img_path (type:string) ローカルファイルの画像パス名
    :return img (type:numpy.ndarray) shape=(H,W,C) 画像データ
    """
    if mode == "net":
        # インターネットから画像をダウンロード
        urllib.request.urlretrieve(img_url, "./lenna.jpg")
        # 画像を読み込み
        img = cv2.imread("./lenna.jpg")  # (256,256,3)

    elif mode == "local":
        img = cv2.imread(img_path)

    return img


def displayImg(img):
    """
    画像を表示
    :param img (type:numpy.ndarray) 画像情報
    """
    cv2.imshow("display image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def displayCenter(centerlist, img):
    """
    クラスタの重心座標を描画
    :param centerlist (type:list) 複数の中心座標を記録したリスト
    :param img (type:numpy.ndarray) 画像情報
    """
    for i in range(len(centerlist)):
        x, y = centerlist[i][-2:]
        cv2.circle(img, (x, y), 5, (0, 0, 255), thickness=-1)
        cv2.putText(
            img,
            text="({},{})".format(x, y),
            org=(x, y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.4,
            color=(0, 0, 0),
        )
    displayImg(img)


def displayCluster(centerlist, label, img):
    """
    各クラスタに含まれる画素をクラスタの重心の色に変更
    :param centerlist (type:list) 各クラスタ重心座標を記録したリスト
    :param label (type:list) 画素がどのクラスタに属しているかを記録したリスト
    :param img (type:numpy.ndarray) 画像情報
    """
    for i in range(len(centerlist)):
        # i番目のクラスタのlab値を取得
        l, a, b = centerlist[i][:3]

        # i番目のクラスタに含まれる点のxy座標 [0]: y座標, [1]:x座標
        idxs = np.where(label == i)

        # 座標情報を[y,x]の形式に変更
        idxs = np.stack([idxs[0], idxs[1]], 1)

        # 各座標の値をクラスタ重心のlab値に変更
        img[tuple((idxs).T)] = [l, a, b]

    # Lab->BGRは、BGR->Labの逆変換で可能。省略のため、今回はOpenCVにより実行
    img = cv2.cvtColor(img, cv2.COLOR_Lab2BGR)
    displayImg(img)


def displayClusterColor(centerlist, label, img):
    """
    各クラスタに含まれる画素をクラスタの重心の色に変更
    :param centerlist (type:list) 各クラスタ重心座標を記録したリスト
    :param label (type:list) 画素がどのクラスタに属しているかを記録したリスト
    :param img (type:numpy.ndarray) 画像情報
    """
    for i in range(len(centerlist)):
        # i番目のクラスタに含まれる点のxy座標 [0]: y座標, [1]:x座標
        idxs = np.where(label == i)

        # 座標情報を[y,x]の形式に変更
        idxs = np.stack([idxs[0], idxs[1]], 1)

        # 各座標の値をクラスタ重心のlab値に変更
        img[tuple((idxs).T)] = list(np.random.choice(range(256), size=3))

    for i in range(len(centerlist)):
        x, y = centerlist[i][-2:]
        cv2.circle(img, (x, y), 5, (0, 0, 255), thickness=-1)
    displayImg(img)


def displayClusterContour(label, img):
    """
    各クラスタの輪郭を描画
    :param label (type:list) 画素がどのクラスタに属しているかを記録したリスト
    :param img (type:numpy.ndarray) 画像情報
    """
    h, w = img.shape[:2]
    for y in range(h - 1):
        for x in range(w - 1):
            l1 = label[y][x]
            l2 = label[y + 1][x]
            l3 = label[y][x + 1]
            if not l1 == l2:
                img[y + 1][x] = [0, 255, 255]
            if not l1 == l3:
                img[y][x + 1] = [0, 255, 255]

    displayImg(img)


def srgb2rgbPixels(srgb):
    """
    ピクセル値を非線形RGBから線形RGBに変換
    :param srgb 非線形RGB値
    :return rgb 線形RGB値
    """
    rgb = []
    for i in range(len(srgb)):
        value = srgb[i]
        value = float(value) / 255
        if value > 0.04045:
            value = ((value + 0.055) / 1.055) ** 2.4
        else:
            value /= 12.92
        value *= 100
        rgb.append(value)
    return rgb


# RGBからXYZへの変換行列
xyz_matrix = [
    [0.412453, 0.357580, 0.180423],
    [0.212671, 0.715160, 0.072169],
    [0.019334, 0.119193, 0.950227],
]


def rgb2xyzPixels(rgb):
    """
    ピクセル値をRGB表色系からXYZ表色系に変換
    :param rgb RGB表色系
    :return xyz XYZ表色系
    """
    [x, y, z] = np.dot(xyz_matrix, rgb)
    x = round(x, 4)
    y = round(y, 4)
    z = round(z, 4)
    xyz = [x, y, z]
    return xyz


def xyz2labPixels(xyz):
    """
    ピクセル値をXYZ表色系からLab表色系に変換
    :param xyz XYZ表色系
    :return lab Lab表色系
    """
    [x, y, z] = xyz
    x = float(x) / 95.0456
    y = float(y) / 100.0
    z = float(z) / 108.8754

    def f(v):
        if v > 0.008856:
            v = v ** (1 / 3)
        else:
            v = 7.787 * v + 16 / 116
        return v

    if y > 0.008856:
        l = (116 * f(y)) - 16
    else:
        l = 903.3 * y

    a = 500 * (f(x) - f(y))
    b = 200 * (f(y) - f(z))
    l = l * 255 / 100
    a += 128
    b += 128
    l = round(l)
    a = round(a)
    b = round(b)
    return [l, a, b]


def bgr2labPixels(bgr):
    """
    ピクセルのBGR値をLab値に変換
    :param bgr (type:numpy.ndarray) shape=(C,) BGRピクセル値
    :return lab (list) length:3 Labピクセル値
    """
    # BGRからRGBに変換
    rgb = bgr[::-1]  # 順番を反転(B->G->R => R->G->B)

    # 非線形RGBから線形RGBに変換
    rgb = srgb2rgbPixels(rgb)

    # RGB表色系からXYZ表色系に変換
    xyz = rgb2xyzPixels(rgb)

    # XYZ表色系からLab表色系に変換
    lab = xyz2labPixels(xyz)

    return lab


def bgr2labImg(img):
    """
    BGR画像をLab画像に変換
    :param img (type:numpy.ndarray) shape=(H,W,C) BGR画像
    :return labimg (type:numpy.ndarray) shape=(H,W,C) Lab画像
    """
    h, w = img.shape[:2]

    # 並列処理準備
    cpu_num = mp.cpu_count()  # 並列処理数
    p = mp.Pool(cpu_num)

    # Numpy配列の形状を変更
    img = img.reshape([-1, 3])

    # RGB画像をLab画像に変換 (並列処理)
    labimg = []
    for result in p.map(bgr2labPixels, img):
        labimg.append(result)

    # int型に変更
    labimg = np.array(labimg, dtype="uint8")

    # Numpy配列の形状を元に戻す
    labimg = labimg.reshape([h, w, 3])

    return labimg
