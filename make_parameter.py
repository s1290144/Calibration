import cv2
import numpy as np

# 画像の読み込み
image = cv2.imread('QR_plant1.png')
h, w = image.shape[:2]

# カメラ行列（仮の値、適宜変更してください）
camera_matrix = np.array([[w, 0, w / 2],
                          [0, w, h / 2],
                          [0, 0, 1]], dtype=np.float32)

# 歪み係数の初期値
dist_coeffs_org = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

# QRコードの検出およびデコード関数
def detect_qr_code(image):
    # OpenCVのQRコード検出器
    qr_detector = cv2.QRCodeDetector()

    # QRコードのデコードと位置取得
    data, points, _ = qr_detector.detectAndDecode(image)

    if data:  # デコード成功時
        points = points[0] if points is not None else None
        return data, points
    else:
        return None, None

# 歪み補正を行う関数
def undistort_image(image, dist_coeffs):
    # カメラ行列と歪み係数からマップを生成
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    # 歪みマップを生成
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1)
    mapx_org, mapy_org = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs_org, None, new_camera_matrix, (w, h), cv2.CV_32FC1)

    # 歪み補正を適用
    undistorted_image = cv2.remap(image, mapx, mapy_org, interpolation=cv2.INTER_LINEAR)
    return undistorted_image

# パラメータの範囲とステップ数を定義
k1_range = np.linspace(-0.1, 0.1, num=21)
k2_range = np.linspace(-0.1, 0.1, num=21)
p1_range = np.linspace(-0.1, 0.1, num=21)
p2_range = np.linspace(-0.1, 0.1, num=21)
k3_range = np.linspace(-0.1, 0.1, num=21)

# ファイルを開いて書き込む準備
with open("parameter.txt", "w") as f:
    # ループでパラメータを変化させて歪み補正を行う
    for k1 in k1_range:
        for k2 in k2_range:
            for p1 in p1_range:
                for p2 in p2_range:
                    for k3 in k3_range:
                        # 歪み係数を更新
                        dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

                        # 歪み補正を適用した画像を取得
                        corrected_image = undistort_image(image, dist_coeffs)

                        # QRコードを検出およびデコード
                        data, qr_points = detect_qr_code(corrected_image)

                        if data is not None:  # QRコードの内容が読み取れた場合のみ処理
                            # パラメータの値をファイルに書き込む
                            f.write(f"{k1} {k2} {p1} {p2} {k3}\n")
