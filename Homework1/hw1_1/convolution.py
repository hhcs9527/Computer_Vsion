import numpy as np 
import cv2
def zero_padding(M, filters):
    add = int(filters.shape[0]/2)
    pad_result = np.zeros((M.shape[0]+2*add, M.shape[1]+2*add))
    pad_result[add:pad_result.shape[1]-add, add:pad_result.shape[1]-add] = M

    return pad_result

def convolution2D(M, filters):
    pad_result = zero_padding(M, filters)
    L = M.shape[0]
    add = int(filters.shape[0]/2)
    result_pic = np.ones(M.shape)

    for i in range(L):
        for j in range(L):
            # convolution center is (add+i:, add+j)
            result_pic[i][j] = np.sum(pad_result[i:i+2*add+1, j: j+2*add+1]*filters)

    return result_pic



def g_zero_padding(M, filters):
    add = int(filters.shape[0]/2)

    pad_result = np.zeros((M.shape[0]+2*add, M.shape[1]+2*add, M.shape[2]))
    pad_result[add:pad_result.shape[1]-add, add:pad_result.shape[1]-add,:] = M

    return pad_result


def guassian_convolution2D(M, filters):
    pad_result = g_zero_padding(M, filters)
    L = M.shape[0]
    add = int(filters.shape[0]/2)
    result_pic = np.ones(M.shape)
    for k in range(M.shape[2]):
        for i in range(L):
            for j in range(L):
                # convolution center is (add+i:, add+j)
                if filters.shape[0] %2 == 0:
                    result_pic[i][j][k] = np.sum(pad_result[i:i+2*add, j: j+2*add, k]*filters)
                else:
                    result_pic[i][j][k] = np.sum(pad_result[i:i+2*add+1, j: j+2*add+1, k]*filters)

    return result_pic



if __name__ == "__main__":
    img = cv2.imread('original.jpg')
    filters = np.ones((3,3))

    convolution2D(img, filters)