import numpy as np 
import cv2
import time


class Two_screen():
    def __init__(self, load, pic, kernel_size): 
        self.load_npy = load
        self.read_pic = pic       
        self.read_file()
        self.kernel_size = kernel_size
        self.img = cv2.imread(self.read_pic)
        


    def read_file(self):
        data = np.load(self.load_npy)
        self.input_data = np.array(data[:4])
        self.target_data = np.array(data[4:])


    def poly_determine(self, m, boundary):
        A = boundary[2]
        B = boundary[0]
        C = boundary[1]
        D = boundary[3]
        a = (B[0] - A[0])*(m[1] - A[1]) - (B[1] - A[1])*(m[0] - A[0])
        b = (C[0] - B[0])*(m[1] - B[1]) - (C[1] - B[1])*(m[0] - B[0])
        c = (D[0] - C[0])*(m[1] - C[1]) - (D[1] - C[1])*(m[0] - C[0])
        d = (A[0] - D[0])*(m[1] - D[1]) - (A[1] - D[1])*(m[0] - D[0])
        if((a > 0 and b > 0 and c > 0 and d > 0) or (a < 0 and b < 0 and c < 0 and d < 0)): 
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


    def smooth(self, i, j):
        filter_one = np.ones((self.kernel_size, self.kernel_size))
        L = int((self.kernel_size-1)/2)
        filter_one[L, L] = 0
        mask = self.mask[j-L:j+L+1, i-L:i+L+1]
        mask[mask > 1]= 0
        total_sum = mask.reshape(1,-1).squeeze().tolist().count(1)

        new_assign = ((filter_one*mask).reshape(self.kernel_size,self.kernel_size,-1))*self.towrite[j-L:j+L+1, i-L:i+L+1]

        self.towrite[j,i] = np.sum(np.sum(new_assign, axis = 1), axis = 0)/total_sum
        self.mask[j,i] = 1

    
    def prepare_parameter(self):

        self.h, self.w,_= self.img.shape
        self.pointL = np.zeros((self.h, self.w))
        self.pointR = np.zeros((self.h, self.w))
        self.mask = np.zeros((self.h, self.w))+2 

        self.L2R = self.solve_homography(self.input_data, self.target_data)
        self.R2L = self.solve_homography(self.target_data, self.input_data)


    def find_point(self):

        # store in the img (x,y) but for is matrix (y,x)
        for i in range(self.w):
            for j in range(self.h):
                p = np.array([i,j])
                if (self.poly_determine(p, self.input_data)):
                    self.pointL[j,i] = 1
                    self.mask[j,i] = 0
                if (self.poly_determine(p, self.target_data)):
                    self.pointR[j,i] = 1
                    self.mask[j,i] = 0


        # get who to transfer
        x,y = np.where(self.pointL == 1)
        self.Left_screen = np.insert(list(zip(y,x)), 2, 1, axis = 1)

        x,y = np.where(self.pointR == 1)
        self.Right_screen = np.insert(list(zip(y,x)), 2, 1, axis = 1)


    def Forwarding(self):   

        print('Two screen exchange Forwarding...')
        self.towrite = cv2.imread(self.read_pic)

        # transfer to somewhere
        # store in the img (x,y) but for is matrix (y,x)

        for i in self.Left_screen:
            x, y, z = self.L2R @ i
            x = int(x / z)
            y = int(y / z)
            self.towrite[y,x] = self.img[i[1], i[0]]
            self.mask[y,x] = 1

        for i in self.Right_screen:
            x, y, z = self.R2L @ i
            x = int(x / z)
            y = int(y / z)
            self.towrite[y,x] = self.img[i[1], i[0]]
            self.mask[y,x] = 1         

        x, y = np.where(self.mask == 0)
        need_to_fix = np.array(list(zip(y,x)))

        for i in need_to_fix:
            self.smooth(i[0], i[1])

        name = './result/two_screen_exchange_'+ 'Forwarding' +'.jpg'
        cv2.imwrite(name, self.towrite)


    def Backwarding(self):  

        print('Two screen exchange Backwarding...')
        self.towrite = cv2.imread(self.read_pic)

        # transfer to somewhere
        # store in the img (x,y) but for is matrix (y,x)

        L2R = np.linalg.inv(self.L2R)
        R2L = np.linalg.inv(self.R2L)

        for i in self.Right_screen:
            x, y, z = L2R @ i
            x = int(x / z)
            y = int(y / z)
            self.towrite[i[1], i[0]] = self.img[y,x]

        for i in self.Left_screen:
            x, y, z = R2L @ i
            x = int(x / z)
            y = int(y / z)
            self.towrite[i[1], i[0]] = self.img[y,x]

        name = './result/two_screen_exchange_'+ 'Backwarding' +'.jpg'
        cv2.imwrite(name, self.towrite)


    def plot_edge(self):
        # 畫線用的 function
        point_color = (0, 255, 0) # BGR
        point_color1 = (0, 0, 255) # BGR
        thickness = 8 # 可以 0 、4、8
        cv2.line(self.img, tuple(self.input_data[0]), tuple(self.input_data[1]), point_color, thickness)
        cv2.line(self.img, tuple(self.input_data[0]), tuple(self.input_data[2]), point_color, thickness)
        cv2.line(self.img, tuple(self.input_data[1]), tuple(self.input_data[3]), point_color, thickness)
        cv2.line(self.img, tuple(self.input_data[2]), tuple(self.input_data[3]), point_color, thickness)

        cv2.line(self.img, tuple(self.target_data[0]), tuple(self.target_data[1]), point_color1, thickness)
        cv2.line(self.img, tuple(self.target_data[0]), tuple(self.target_data[2]), point_color1, thickness)
        cv2.line(self.img, tuple(self.target_data[1]), tuple(self.target_data[3]), point_color1, thickness)
        cv2.line(self.img, tuple(self.target_data[2]), tuple(self.target_data[3]), point_color1, thickness)

        cv2.imwrite('./result/two_screen_exchange_edge.jpg', self.img)


    def screen_exchange(self):
        self.prepare_parameter()
        self.find_point()
        self.Forwarding()
        self.Backwarding()
        self.plot_edge()


if __name__ == '__main__':
    s = time.time()      
    #H2 = HW2_2_two_screen('./npy/test_1.npy', './data/test_img.png', 5, False)
    H2 = Two_screen('./npy/two_screen.npy', './data/two_screen.jpg', 5)
    H2.screen_exchange()
    e = time.time()
    print('spend ', e-s)
