import cv2
import numpy as np

# 画像の読み込み
image = cv2.imread('QR_plant1.png')

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
    h, w = image.shape[:2]
    # カメラ行列（仮の値、適宜変更してください）
    camera_matrix = np.array([[w, 0, w / 2],
                              [0, w, h / 2],
                              [0, 0, 1]], dtype=np.float32)
    # カメラ行列と歪み係数からマップを生成
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    # 歪みマップを生成
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1)
    mapx_org, mapy_org = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs_org, None, new_camera_matrix, (w, h), cv2.CV_32FC1)
    # 歪み補正を適用
    undistorted_image = cv2.remap(image, mapx, mapy_org, interpolation=cv2.INTER_LINEAR)
    return undistorted_image

# パラメータファイルから値を読み取る関数
def read_parameters_from_file(file_path):
    parameters = []
    with open(file_path, 'r') as f:
        for line in f:
            k1, k2, p1, p2, k3 = map(float, line.split())
            parameters.append((k1, k2, p1, p2, k3))
    return parameters

def calibration(image):
    # パラメータファイルのパス
    parameter_file = 'parameter.txt'
    parameter_list = read_parameters_from_file(parameter_file)

    # パラメータリストを使用してQRコードの検出とデコードを試行
    for params in parameter_list:
        k1, k2, p1, p2, k3 = params
        # 歪み係数を設定
        dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
        # 歪み補正を適用
        corrected_image = undistort_image(image, dist_coeffs)
        # QRコードを検出およびデコード
        data, qr_points = detect_qr_code(corrected_image)
        if data is not None:  # QRコードの内容が読み取れた場合のみ出力       
            image = corrected_image
            #cv2.putText(image, data, org=(100, 300), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_4)
            break
    #print("みっけた")
    return image

image = calibration(image)
cv2.imshow('Undistorted Image', image) #imageを上書き
cv2.waitKey(0)
