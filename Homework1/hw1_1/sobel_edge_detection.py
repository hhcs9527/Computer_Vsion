import cv2
import numpy as np
import matplotlib.pyplot as plt
import gaussian_smooth
import convolution as con

def get_sobel(img):
    # make it into 64 bit
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    Hx = np.array([[-1,0,1], 
                    [-2,0,2], 
                    [-1,0,1]], dtype = np.float32)/8

    Hy = np.array([[1,2,1], 
                    [0,0,0], 
                    [-1,-2,-1]], dtype = np.float32)/8

    x_d = con.convolution2D(img_grey, Hx) #cv2.filter2D(img_grey, -1, Hx)
    y_d = con.convolution2D(img_grey, Hy) #cv2.filter2D(img_grey, -1, Hy)
    
    return x_d, y_d



def sobel_edge_detection(img, kernel_size):
    
    x_d, y_d = get_sobel(img)
    
    # get direction matrix, by arctan(y_d/x_d)
    gradient = np.arctan2(y_d, x_d).astype(np.uint8)
    magnitude = np.sqrt(x_d*x_d + y_d*y_d).astype(np.uint8)

    return x_d, y_d, gradient, magnitude



if __name__ == "__main__":
    # where workflow should do
    path = 'Gaussian_smooth_kernel_size(10).jpg' 
    img = cv2.imread(path)
    kernel_size = 10
    x_d, y_d, gradient, magnitude = sobel_edge_detection(img, kernel_size)

    # Save result
    cv2.imwrite('sobel_xd_kernel_size('+ str(kernel_size) + ').jpg', x_d)
    cv2.imwrite('sobel_yd_kernel_size('+ str(kernel_size) + ').jpg', y_d)
    cv2.imwrite('gradient_kernel_size('+ str(kernel_size) + ').jpg', gradient)
    cv2.imwrite('magnitude_kernel_size('+ str(kernel_size) + ').jpg', magnitude)
    
    plt.imshow(gradient, cmap = plt.cm.summer)
    plt.savefig('gradient_cmap_kernel_size('+ str(kernel_size) + ').jpg')