"""
https://qiita.com/sitar-harmonics/items/ba02cd14d6f362439e96
"""

import cv2


def main():
    image = cv2.imread("lenna.jpg")

    # パラメータ
    algorithms = [
        ("SLIC", cv2.ximgproc.SLIC),
        ("SLICO", cv2.ximgproc.SLICO),
        ("MSLIC", cv2.ximgproc.MSLIC),
    ]

    region_size = 20
    ruler = 30
    min_element_size = 10
    num_iterations = 4

    # BGR-HSV変換
    converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    for alg in algorithms:
        slic = cv2.ximgproc.createSuperpixelSLIC(
            converted, alg[1], region_size, float(ruler)
        )
        slic.iterate(num_iterations)
        slic.enforceLabelConnectivity(min_element_size)
        result = image.copy()

        # スーパーピクセルセグメンテーションの境界を取得
        contour_mask = slic.getLabelContourMask(False)
        result[0 < contour_mask] = (0, 255, 255)
        cv2.imshow("SLIC (" + alg[0] + ") result", result)
        # cv2.imwrite('images/SLIC_'+alg[0]+'_result.jpg', result)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
