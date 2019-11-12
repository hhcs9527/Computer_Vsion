from Two_pic import Two_pic
from Two_screen import Two_screen
import time


if __name__ == "__main__":
    s = time.time()  
    HW_2_2_1 = Two_screen('./npy/two_screen.npy', './data/two_screen.jpg', 5)
    HW_2_2_2 = Two_pic('./npy/ntu.npy', './npy/road.npy', './data/ntu.jpg', './data/road.jpg', 5)

    HW_2_2_1.screen_exchange()
    HW_2_2_2.pic_exchange()    
    e = time.time()
    print('spend ', e-s)