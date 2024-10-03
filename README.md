# Calibration

## make_parameter.py
QR_plant1.pngをパラメータをループで変えながら歪み補正を行い、QRコードの内容が読み取れたパラメータ値をparameter.txtに行別に書き込むコードです。

## calibration.py
parameter.txtのパラメータ値をループで試し、QRコードの内容がわかったらbreakし、内容がわかった画像をimageに上書きするコードです。

