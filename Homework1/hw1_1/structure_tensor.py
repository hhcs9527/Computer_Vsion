import sobel_edge_detection as so
import matplotlib.pyplot as plt
import numpy as np 
import cv2
import convolution as con

def nms_compare(x_d, y_d, angle, gradient, i, j):
    #q ,r = 0, 0
    # angel 0
    if (0 <= angle[i][j] < 22.5) or (337.5 <= angle[i][j] <= 360) or (157.5 <= angle[i][j] < 202.5):
        q = gradient[i][j+1]
        r = gradient[i][j-1]

    # angle 45
    elif (22.5 <= angle[i][j] < 67.5) or (202.5 <= angle[i][j] < 247.5):
        q = gradient[i+1][j-1]
        r = gradient[i-1][j+1]

    # angle 90
    elif (67.5 <= angle[i][j] < 112.5) or (247.5 <= angle[i][j] < 292.5):
        q = gradient[i+1][j]
        r = gradient[i-1][j]

    # angle 135
    elif (112.5 <= angle[i][j] < 157.5) or (292.5 <= angle[i][j] < 337.5):
        q = gradient[i-1][j-1]
        r = gradient[i+1][j+1]

    if not((gradient[i][j] >= q) and (gradient[i][j] >= r)):
        y_d[i][j] = 0
        x_d[i][j] = 0

    return x_d[i][j], y_d[i][j]


def non_max_subpression(x_d, y_d):
    gradient = np.arctan2(y_d, x_d)
    # change to angle and classify into 4 classes, 0, 45, 90, 135
    angle = gradient * 180. / np.pi
    l0, l1 = angle.shape
    angle[angle < 0] += 360


    for i in range(1, l0-1):
        for j in range(1, l1-1):
            x_d[i][j], y_d[i][j] = nms_compare(x_d, y_d, angle, gradient, i, j)

    return x_d, y_d


def Harris_Response(x_d, y_d, k, window_size):
    window = np.ones((window_size, window_size), dtype = float)

    Sxx = con.convolution2D(x_d*x_d, window)#cv2.filter2D(x_d*x_d, -1, window)
    Sxy = con.convolution2D(x_d*y_d, window)#cv2.filter2D(x_d*y_d, -1, window)
    Syy = con.convolution2D(y_d*y_d, window)#cv2.filter2D(y_d*y_d, -1, window)

    Harris_Response = (Sxx*Syy - Sxy*Sxy) - k*(Sxx + Syy)*(Sxx + Syy)

    return Harris_Response



# By definition, in Response
# Edge : r < 0 Corner : r > 0 Flat: r = 0
def type_distinguish(Response, img):
    img_result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)

    img_result[Response > 0.0001*Response.max()] = 255
    return img_result


def structure_tensor(x_d, y_d, k, img, window_size):

    x_d, y_d = non_max_subpression(x_d, y_d)
    Response = Harris_Response(x_d, y_d, k, window_size)
    
    img = type_distinguish(Response, img)

    return img



if __name__ == "__main__":
    # the workfolw should go like this
    # where workflow should do
    path = 'Gaussian_smooth_kernel_size(10).jpg' 
    img = cv2.imread(path)
    kernel_size = 10
    x_d, y_d, gradient, magnitude = so.sobel_edge_detection(img, kernel_size)
    k = 0.04
    window_size = 3
    img = structure_tensor(x_d, y_d, k, img, window_size)

    cv2.imwrite('Test.jpg', img)