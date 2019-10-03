import cv2
import numpy as np 
import math

def Guassian(x, y, var):
    return (1/(2*math.pi*var*var))*math.exp(-(x*x+y*y)/(2*var*var))

def Guassian_kernel(size, var):
    kernel = np.ones((size,size), float)
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

def gaussian_smooth(size, var):
    # Read image
    img = cv2.imread('original.jpg')

    # get Guassian Kernel
    kernel = Guassian_kernel(size, var)
    result = cv2.filter2D(img, -1, kernel, anchor=(-1,-1))

    cv2.imwrite('Gaussian_smooth_kernel_size('+ str(size)+').jpg', result)


if __name__ == "__main__":
    size = 10
    var = 5
    gaussian_smooth(size, var)