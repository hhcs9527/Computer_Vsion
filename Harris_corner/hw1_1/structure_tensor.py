import sobel_edge_detection as so
import matplotlib.pyplot as plt
import numpy as np 
import cv2
import convolution as con
import gaussian_smooth as gu
import scipy.ndimage as ndi

def non_maximum_supression(Harris_Response, threshold, window_size):
    W,H = Harris_Response.shape
    corners = []

    skip = (Harris_Response <= threshold)   # True skip, False not to skip

    for i in range(window_size-1 , W-window_size):
        j = window_size
        while j < H - window_size and (skip[i,j] or Harris_Response[i,j-1] >= Harris_Response[i,j]):
            j = j + 1

        while j < H - window_size:
            while j < H - window_size and (skip[i,j] or Harris_Response[i,j+1]>= Harris_Response[i,j]):
                j += 1

            if j < H - window_size:
                p1 = j + 2

                while p1 <= j + window_size and Harris_Response[i, p1] < Harris_Response[i,j]:
                    skip[i, p1] = True
                    p1 = p1 + 1

                if p1 > j+ window_size :
                    p2 = j-1
                    while p2 >= j - window_size and Harris_Response[i, p2] <= Harris_Response[i,j]:
                        p2 = p2 - 1

                    if p2 < j - window_size:
                        k = i + window_size
                        found = False
                        
                        while not found and k > i:
                            l = j + window_size
                            while not found and l >= j - window_size:
                                if Harris_Response[k,l] > Harris_Response[i,j]:
                                    found = True
                                else:
                                    skip[k,l] = True
                                l = l - 1
                            k = k - 1
                        k = i - window_size

                        while not found and k < i:
                            l = j - window_size
                            while not found and l <= j + window_size:
                                if Harris_Response[k,l] >= Harris_Response[i,j]:
                                    found = True
                                l = l + 1
                            k += 1
                        
                        if not found :
                            corners.append((j,i))
                j = p1
    
    return corners



def Harris_Response(x_d, y_d, k, window_size):
    window = gu.Guassian_kernel(window_size, 5)#np.ones((3,3))

    Sxx = cv2.filter2D(x_d*x_d, -1, window) #con.convolution2D(x_d*x_d, window) #cv2.filter2D(x_d*x_d, -1, window)
    Sxy = cv2.filter2D(x_d*y_d, -1, window) #con.convolution2D(x_d*y_d, window) #cv2.filter2D(x_d*y_d, -1, window)
    Syy = cv2.filter2D(y_d*y_d, -1, window) #con.convolution2D(y_d*y_d, window) #cv2.filter2D(y_d*y_d, -1, window)

    Harris_Response = (Sxx*Syy - Sxy*Sxy) - k*(Sxx + Syy)*(Sxx + Syy)

    threshold = 0.01*Harris_Response.max()
    corners  = non_maximum_supression(Harris_Response, threshold, 5)  

    return corners



# By definition, in Response
# Edge : r < 0 Corner : r > 0 Flat: r = 0
def type_distinguish(corners, img):
    img_result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_return = cv2.cvtColor(img_result, cv2.COLOR_GRAY2BGR)

    point_size = 2
    point_color = (0, 0, 255) # BGR
    thickness = 4 # 可以为 0 、4、8

    for point in corners:
        cv2.circle(img_return, point, point_size, point_color, thickness)

    return img_return



def structure_tensor(x_d, y_d, k, img, window_size):

    corners = Harris_Response(x_d, y_d, k, window_size)
    
    img = type_distinguish(corners, img)

    return img





if __name__ == "__main__":
    # the workfolw should go like this
    # where workflow should do
    path = 'Gaussian_smooth_kernel_size(10).jpg' 
    img = cv2.imread(path)
    kernel_size = 10
    x_d, y_d, gradient, magnitude = so.sobel_edge_detection(img, kernel_size)
    k = 0.04
    window_size = 30
    img = structure_tensor(x_d, y_d, k, img, window_size)
    cv2.imwrite('Structure_tensor_window_size('+ str(window_size) + ').jpg', img)
    

