
import numpy as np

from demosaic_2004 import demosaicing_CFA_Bayer_Malvar2004


def mosaic(img, pattern):
    '''
    Input:
        img: H*W*3 numpy array, input image.
        pattern: string, 4 different Bayer patterns (GRBG, RGGB, GBRG, BGGR)
    Output:
        output: H*W numpy array, output image after mosaic.
    '''
    ########################################################################
    # TODO:                                                                #
    #   1. Create the H*W output numpy array.                              #   
    #   2. Discard other two channels from input 3-channel image according #
    #      to given Bayer pattern.                                         #
    #                                                                      #
    #   e.g. If Bayer pattern now is BGGR, for the upper left pixel from   #
    #        each four-pixel square, we should discard R and G channel     #
    #        and keep B channel of input image.                            #     
    #        (since upper left pixel is B in BGGR bayer pattern)           #
    ########################################################################

    # input image 3 channel -> Red, Green, Blue
    # 因為mosiac過程類只因為pattern不同所以放入方式不同, 先判斷格式, 再做

    if pattern == "BGGR":
        pattern_code = [0,1,0,2]

    elif pattern == "RGGB":
        pattern_code = [0,0,0,2]

    elif pattern == "GBRG":
        pattern_code = [1,1,0,2]
 
    elif pattern == "GRBG":
        pattern_code = [1,0,0,2]
        
    # Pattern_Code meaning, 
    # Pattern_Code[0] find the position to find the R/B
    # Pattern_Code[1] 決定哪邊要放 R
    # Pattern_Code[2] 放 R
    # Pattern_Code[3] 放 B

    H,W,D = img.shape

    output = img[:,:,1]
    for i in range(H):
        for j in range(W):
            if (i+j)%2 == pattern_code[0]:
                if i%2 == pattern_code[1]:
                    output[i,j] = img[i,j,pattern_code[2]]
                else:
                    output[i,j] = img[i,j,pattern_code[3]]

    ########################################################################
    #                                                                      #
    #                           End of your code                           #
    #                                                                      # 
    ########################################################################

    return output


def demosaic(img, pattern):
    '''
    Input:
        img: H*W numpy array, input RAW image.
        pattern: string, 4 different Bayer patterns (GRBG, RGGB, GBRG, BGGR)
    Output:
        output: H*W*3 numpy array, output de-mosaic image.
    '''
    #### Using Python colour_demosaicing library
    #### You can write your own version, too
    output = demosaicing_CFA_Bayer_Malvar2004(img, pattern)
    output = np.clip(output, 0, 1)

    return output

