#  hw1_1 的程式流程

## 主要分成 4 個.pypy 檔,
分別是 hw1_1, gaussian_smooth, sobel_edge_detection, structure_tensor.py

## hw1_1
控制所有流程,其結果為一次做完 smooth, sobel, structure tensor,
其中可以用 choose 來決定那一種 input image(0 一般, 1 旋轉 30 度, 2 縮放 0.5))

## gaussian_smooth
做 gaussian smooth 的副程式,首先會先生成一個跟 kernel 同 size 的 matrix,再以中心為原點帶入 gaussian 的公式得值,最後再 normalize,並利用 cv2.pyfilter2D 完成 convolution,已達到 gaussian smooth 的結果,再利用 cv2 畫出結果

## sobel_edge_detection
沿用 gaussian smooth 的結果繼續處理,從 function get sobel 得到 sobel operator做 convolution 得到 x_d, y_d,再根據定義得到 magnitude(小於 3 者設為 0),gradient,再利用 cv2 畫出結果

## structure_tensor
沿用 sobel_edge_detection 的結果繼續處理,先計算 Ix*Ix, Ix*Iy, Iy*Iy,並以np.pyones((3,3))當作 window finction,對 Ix*Ix, Ix*Iy, Iy*Iy 都做 convolution 得到 Sxx, Sxy, Syy.py. 

所以Harris Response = (Sxx * Syy – Sxy*Sxy) - k*(Sxx + Syy)*(Sxx + Syy),並利用 function non-maximum-subpression,得到 local max 的 corner 座標,再利用 cv2 畫出結果

## convolution.pypy ( 後來沒用 )
在助教未說明是否可以用 lib 時手刻 convolution 的結果

## 會分成 4 個檔案的原因:
1.py 使整體過程明確
2.py 能分批達到作業要求/一次完成流程(旋轉/縮放)
