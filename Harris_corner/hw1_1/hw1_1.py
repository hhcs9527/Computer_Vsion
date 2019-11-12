import gaussian_smooth as gu
import sobel_edge_detection as sob
import structure_tensor as st
import cv2

def Harris_Corner_Detection(path, var, kernel_size, window_size, k, choose):

    smooth_result = gu.gaussian_smooth(path, kernel_size, var, choose)

    x_d, y_d, gradient, magnitude = sob.sobel_edge_detection(smooth_result, kernel_size)

    img_result = st.structure_tensor(x_d, y_d, k, smooth_result, window_size)
    
    return img_result


if __name__ == "__main__":
    path = 'original.jpg'
    kernel_size = 10
    var = 5
    window_size = 3
    k = 0.04
    choose = 1

    img = Harris_Corner_Detection(path, var, kernel_size, window_size, k, choose)

    #cv2.imwrite("Scale_kernel(10)_window_size(3).jpg", img)
    cv2.imwrite("Rotate30_kernel(10)_window_size(3).jpg", img)