import cv2
import numpy as np

def get_sobel_operator():
    Hx = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
    Hy = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])
    return Hx/8, Hy/8

def sobel_edge_detection():
    img = cv2.imread('original.jpg')
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    
    Hx, Hy = get_sobel_operator()
    
    x_d = cv2.filter2D(img_grey, -1, Hx)
    y_d = cv2.filter2D(img_grey, -1, Hy)


    cv2.imwrite('sobel_xd.jpg', x_d)
    cv2.imwrite('sobel_yd.jpg', y_d)





if __name__ == "__main__":
    sobel_edge_detection()