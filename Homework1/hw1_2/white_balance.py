
import numpy as np

def generate_wb_mask(img, pattern, fr, fb):
    '''
    Input:
        img: H*W numpy array, RAW image
        pattern: string, 4 different Bayer patterns (GRBG, RGGB, GBRG, BGGR)
        fr: float, white balance factor of red channel
        fb: float, white balance factor of blue channel 
    Output:
        mask: H*W numpy array, white balance mask
    '''
    ########################################################################
    # TODO:                                                                #
    #   1. Create a numpy array with shape of input RAW image.             #
    #   2. According to the given Bayer pattern, fill the fr into          #
    #      correspinding red channel position and fb into correspinding    #
    #      blue channel position. Fill 1 into green channel position       #
    #      otherwise.                                                      #
    ########################################################################
    H,W = img.shape
    mask = np.ones((H,W))

    if pattern == "BGGR":
        pattern_code = [0,1]

    elif pattern == "RGGB":
        pattern_code = [0,0]

    elif pattern == "GBRG":
        pattern_code = [1,1]
 
    elif pattern == "GRBG":
        pattern_code = [1,0]

    # Pattern_Code meaning, 
    # Pattern_Code[0] find the position to find the R/B
    # Pattern_Code[1] 決定哪裡要放 R
    # 依序填入 R, B


    for i in range(H):
        for j in range(W):
            if (i+j)%2 == pattern_code[0]:
                if i%2 == pattern_code[1]:
                    mask[i,j] = fr
                else:
                    mask[i,j] = fb

    ########################################################################
    #                                                                      #
    #                           End of your code                           #
    #                                                                      # 
    ########################################################################
        
    return mask