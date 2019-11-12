import numpy as np 
import cv2
import time


class Two_pic():
    def __init__(self, npy1, npy2, pic1, pic2, kernel_size): 
        self.load_npy1 = npy1
        self.load_npy2 = npy2
        self.read_pic1 = pic1       
        self.read_pic2 = pic2 

        self.img1 = cv2.imread(self.read_pic1)
        self.img2 = cv2.imread(self.read_pic2)
        
        self.read_file()
        self.kernel_size = kernel_size


    def read_file(self):
        # idea : pic1 as input, pic2 as projecting target
        self.input_data = np.load(self.load_npy1)
        self.target_data = np.load(self.load_npy2)

        
    def poly_determine(self, m, boundary):
        A = boundary[2]
        B = boundary[0]
        C = boundary[1]
        D = boundary[3]
        a = (B[0] - A[0])*(m[1] - A[1]) - (B[1] - A[1])*(m[0] - A[0])
        b = (C[0] - B[0])*(m[1] - B[1]) - (C[1] - B[1])*(m[0] - B[0])
        c = (D[0] - C[0])*(m[1] - C[1]) - (D[1] - C[1])*(m[0] - C[0])
        d = (A[0] - D[0])*(m[1] - D[1]) - (A[1] - D[1])*(m[0] - D[0])
        if((a >= 0 and b >= 0 and c >= 0 and d >= 0) or (a <= 0 and b <= 0 and c <= 0 and d <= 0)): 
            return True
        return False


    def solve_homography(self, input_data, target_data):
        A = []
        for i in range(len(input_data)):
            A.append([input_data[i,0], input_data[i,1], 1, 0 ,0 ,0, -target_data[i,0]*input_data[i,0], -target_data[i,0]*input_data[i,1], -target_data[i,0]])
            A.append([0, 0, 0, input_data[i,0], input_data[i,1], 1, -target_data[i,1]*input_data[i,0], -target_data[i,1]*input_data[i,1], -target_data[i,1]])
        
        A = np.array(A)
        _, _, vh = np.linalg.svd(A, full_matrices=True)

        V = np.transpose(vh)
        H = (V[:,-1]).reshape(3,3)

        return H


    def smooth(self, i, j, number):
        filter_one = np.ones((self.kernel_size, self.kernel_size))
        L = int((self.kernel_size-1)/2)

        if number == 1:
            mask = self.mask1[j-L:j+L+1, i-L:i+L+1]
        else:
            mask = self.mask2[j-L:j+L+1, i-L:i+L+1]

        filter_one[L, L] = 0
        mask[mask > 1]= 0

        total_sum = mask.reshape(1,-1).squeeze().tolist().count(1)

        if number == 1:
            new_assign = ((filter_one*mask).reshape(self.kernel_size,self.kernel_size,-1))*self.towrite1[j-L:j+L+1, i-L:i+L+1]
            self.towrite1[j,i] = np.sum(np.sum(new_assign, axis = 1), axis = 0)/total_sum
            self.mask1[j,i] = 1
        
        else:
            new_assign = ((filter_one*mask).reshape(self.kernel_size,self.kernel_size,-1))*self.towrite2[j-L:j+L+1, i-L:i+L+1]
            self.towrite2[j,i] = np.sum(np.sum(new_assign, axis = 1), axis = 0)/total_sum
            self.mask2[j,i] = 1            
        
    
    def prepare_parameter(self):
        # pointL for pic1, pointR for pic2, since two picture is the same size
        self.h, self.w,_= self.img1.shape
        self.pointL = np.zeros((self.h, self.w))
        self.pointR = np.zeros((self.h, self.w))

        self.mask1 = np.zeros((self.h, self.w))+2 
        self.mask2 = np.zeros((self.h, self.w))+2 

        self.L2R = self.solve_homography(self.input_data, self.target_data)
        self.R2L = self.solve_homography(self.target_data, self.input_data)


    def find_point(self):

        # store in the img (x,y) but for is matrix (y,x)
        for i in range(self.w):
            for j in range(self.h):
                p = np.array([i,j])
                if (self.poly_determine(p, self.input_data)):
                    self.pointL[j,i] = 1
                    self.mask1[j,i] = 0

                if (self.poly_determine(p, self.target_data)):
                    self.pointR[j,i] = 1
                    self.mask2[j,i] = 0


        # get who to transfer
        # idea Left pic/screen as pic1, Right pic/screen as pic2
        x,y = np.where(self.pointL == 1)
        self.Left_pic = np.insert(list(zip(y,x)), 2, 1, axis = 1)

        x,y = np.where(self.pointR == 1)
        self.Right_pic = np.insert(list(zip(y,x)), 2, 1, axis = 1)


    def Forwarding(self):   

        print('Two picture exchange Forwarding...')
        self.towrite1 = cv2.imread(self.read_pic1)
        self.towrite2 = cv2.imread(self.read_pic2)

        # transfer to somewhere
        # store in the img (x,y) but for is matrix (y,x)

        for i in self.Left_pic:
            x, y, z = self.L2R @ i
            x = int(x / z)
            y = int(y / z)
            self.towrite2[y,x] = self.img1[i[1], i[0]]
            self.mask2[y,x] = 1

        for i in self.Right_pic:
            x, y, z = self.R2L @ i
            x = int(x / z)
            y = int(y / z)
            self.towrite1[y,x] = self.img2[i[1], i[0]]
            self.mask1[y,x] = 1         


        # smooth for both pics
        x, y = np.where(self.mask1 == 0)
        need_to_fix = np.array(list(zip(y,x)))

        for i in need_to_fix:
            self.smooth(i[0], i[1], 1)


        x, y = np.where(self.mask2 == 0)
        need_to_fix = np.array(list(zip(y,x)))

        for i in need_to_fix:
            self.smooth(i[0], i[1], 2)

        name = './result/two_pic_exchange_pic1_'+ 'Forwarding' +'.jpg'
        cv2.imwrite(name, self.towrite1)

        name = './result/two_pic_exchange_pic2_'+ 'Forwarding' +'.jpg'
        cv2.imwrite(name, self.towrite2)


    def Backwarding(self):  

        print('Two picture exchange Backwarding...')
        self.towrite1 = cv2.imread(self.read_pic1)
        self.towrite2 = cv2.imread(self.read_pic2)

        # transfer to somewhere
        # store in the img (x,y) but for is matrix (y,x)

        L2R = np.linalg.inv(self.L2R)
        R2L = np.linalg.inv(self.R2L)

        for i in self.Right_pic:
            x, y, z = L2R @ i
            x = int(x / z)
            y = int(y / z)
            self.towrite2[i[1], i[0]] = self.img1[y,x]

        for i in self.Left_pic:
            x, y, z = R2L @ i
            x = int(x / z)
            y = int(y / z)
            self.towrite1[i[1], i[0]] = self.img2[y,x]

        name = './result/two_pic_exchange_pic1_'+ 'Backwarding' +'.jpg'
        cv2.imwrite(name, self.towrite1)
        name = './result/two_pic_exchange_pic2_'+ 'Backwarding' +'.jpg'
        cv2.imwrite(name, self.towrite2)


    def plot_edge(self):
        # 畫線用的 function
        point_color = (0, 255, 0) # BGR
        point_color1 = (0, 0, 255) # BGR
        thickness = 8 # 可以 0 、4、8
        cv2.line(self.img1, tuple(self.input_data[0]), tuple(self.input_data[1]), point_color, thickness)
        cv2.line(self.img1, tuple(self.input_data[0]), tuple(self.input_data[2]), point_color, thickness)
        cv2.line(self.img1, tuple(self.input_data[1]), tuple(self.input_data[3]), point_color, thickness)
        cv2.line(self.img1, tuple(self.input_data[2]), tuple(self.input_data[3]), point_color, thickness)

        cv2.line(self.img2, tuple(self.target_data[0]), tuple(self.target_data[1]), point_color1, thickness)
        cv2.line(self.img2, tuple(self.target_data[0]), tuple(self.target_data[2]), point_color1, thickness)
        cv2.line(self.img2, tuple(self.target_data[1]), tuple(self.target_data[3]), point_color1, thickness)
        cv2.line(self.img2, tuple(self.target_data[2]), tuple(self.target_data[3]), point_color1, thickness)

        cv2.imwrite('./result/two_pic_exchange_edge_pic1.jpg', self.img1)
        cv2.imwrite('./result/two_pic_exchange_edge_pic2.jpg', self.img2)

    def pic_exchange(self):
        self.prepare_parameter()
        self.find_point()
        self.Forwarding()
        self.Backwarding()
        self.plot_edge()


if __name__ == '__main__':
    s = time.time()      
    #H2 = HW2_2_two_pic('./npy/test_1.npy', './data/test_img.png', 5, False)
    H2 = Two_pic('./npy/ntu.npy', './npy/road.npy', './data/ntu.jpg', './data/road.jpg', 5)
    H2.pic_exchange()
    e = time.time()
    print('spend ', e-s)
