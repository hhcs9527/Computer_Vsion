import cv2
import numpy as np 
import math
import convolution as con

def Guassian(x, y, var):
    return (1/(2*math.pi*var*var))*math.exp(-(x*x+y*y)/(2*var*var))

def Guassian_kernel(size, var):
    kernel = np.ones((size,size), np.float32)
    for i in range(size):
        for j in range(size):
            # (-1,-1)  (0,-1)  (1,-1)
            # (-1, 0)  (0, 0)  (1, 0)
            # (-1, 1)  (0, 1)  (1,-1)
            # 用i-int(size/2), j-int(size/2)的結果會存成該有的kernel之transpose
            # 所以存kernel[j][i]
            kernel[j][i] = Guassian(i-int(size/2), j-int(size/2),var)
    # 再正規化之
    summation = np.sum(kernel)

    return kernel/summation

def gaussian_smooth(path, size, var, choose):
    # Read image
    img = cv2.imread(path).astype(np.float32)
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    
    # Rotate 30
    if choose == 1:
        M = cv2.getRotationMatrix2D(center, 30, scale = 1.0)
        img_result = cv2.warpAffine(img, M, (h, w))

    # Scale 0.5
    elif  choose == 2:
        M = cv2.getRotationMatrix2D(center, 0, scale = 0.5)
        img_result = cv2.warpAffine(img, M, (h, w))
    else:
        img_result = img
        
    # get Guassian Kernel
    kernel = Guassian_kernel(size, var)
    result = cv2.filter2D(img_result, -1, kernel, anchor=(-1,-1)) # con.guassian_convolution2D(img_result, kernel).astype(np.float32) 
    
    return result
    


if __name__ == "__main__":
    # where workflow should do
    size = 10
    var = 5
    path = 'original.jpg'
    choose = 0
    result = gaussian_smooth(path, size, var, choose)
    cv2.imwrite('Gaussian_smooth_kernel_size('+ str(size)+').jpg', result)