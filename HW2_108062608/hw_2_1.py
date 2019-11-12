import numpy as np 
import scipy.linalg
import cv2
import visualize

class HW1():
    def __init__(self, name):
        self.filename = name
        self.read_file()
        self.get_obeserve_matrix()


    def read_file(self):
        data2D = np.load("./npy/Point2D_"+ str(self.filename) +".npy")
        with open('./data/Point3D.txt', 'r') as f:
            data3D = []
            line = f.readline().split()
            while line:
                data3D.append([int(line[0]), int(line[1]), int(line[2])])
                line = f.readline().split()  
        data3D = np.array(data3D)
        self.data2D = np.insert(data2D, 2, 1, axis = 1)
        self.data3D = np.insert(data3D, 3, 1, axis = 1)
        self.data = data3D




    def get_obeserve_matrix(self):
        data3D = self.data3D
        data2D = self.data2D
        A = []
        for i in range(len(data3D)):
            A.append([data3D[i,0],data3D[i,1],data3D[i,2],1,0,0,0,0,-data3D[i,0]*data2D[i,0],-data3D[i,1]*data2D[i,0],-data3D[i,2]*data2D[i,0],-data2D[i,0]])
            A.append([0,0,0,0,data3D[i,0],data3D[i,1],data3D[i,2],1,-data3D[i,0]*data2D[i,1],-data3D[i,1]*data2D[i,1],-data3D[i,2]*data2D[i,1],-data2D[i,1]])
        
        self.A = np.array(A)


    # HW 1_1
    def solve_Projection(self):
        u, s, vh = np.linalg.svd(self.A, full_matrices=True)

        V = np.transpose(vh)
        self.Projection = (V[:,-1]).reshape(3,4)

        return self.Projection
    

    # HW 1_2
    def solve_K(self):
        r, q = scipy.linalg.rq(self.Projection[:,:3])
        K = r/r[2,2]
        self.alpha = r[2,2]
        # using column aspect to imagine the multiplication
        if K[0,0] < 0:
            K[:,0] = -K[:,0]
            q[0,:] = -q[0,:]

        if K[1,1] < 0:
            K[:,1] = -K[:,1]
            q[1,:] = -q[1,:]

        self.Intrinsic = K
        self.translation = np.linalg.inv(self.Intrinsic).dot(self.Projection[:,3]/self.alpha)
        self.Rotation = q
    
    # HW 1_3
    def Reproject(self):
        extrinsic = np.zeros((4,4))
        extrinsic[3,3] = 1
        extrinsic[:3,:3] = self.Rotation
        extrinsic[:3,3] = self.translation
        project = np.eye(3,4)
        a = np.vstack([self.Rotation, self.translation])
        self.reProject = self.Intrinsic.dot(project.dot(a))

    
    def Paint_on_pic(self):

        reconstruct = np.transpose(self.Projection.dot(np.transpose(self.data3D)))
        reconstruct =  reconstruct / np.transpose(reconstruct[:,-1].reshape(1,-1))

#### point the points on

        img = cv2.imread('data/chessboard_'+ str(self.filename)+'.jpg')

        point_size = 2
        point_color = (0, 0, 255) # BGR
        point_color1 = (0, 255, 0)
        thickness = 4 # 可以 0 、4、8
        reconstruct = reconstruct.astype(int)
        data = self.data2D[:,:2]
        point = reconstruct[:,:2].astype(int)

        for i in range(len(point)):
            # reconstruction point is red
            cv2.circle(img, tuple(point[i]), point_size, point_color, thickness)
            # Original data is Green
            cv2.circle(img, tuple(data[i]), point_size, point_color1, thickness)
        cv2.imwrite('./result/result_from_chessboard_'+ str(self.filename) +'.jpg', img)

        print( "Reconstruct RMS from chessboard_" + str(self.filename) + " error is :",np.sqrt(np.sum((reconstruct-self.data2D)**2)/36) )
    
    def do_homework(self):
        self.solve_Projection()
        self.solve_K()
        self.Reproject()
        self.Paint_on_pic()     
        return self.Rotation, self.translation, self.data






if __name__ == "__main__":
    H1 = HW1(1)
    R1, T1, pts = H1.do_homework()
    H2 = HW1(2)
    R2, T2, pts = H2.do_homework()
    visualize.visualize(pts, R1, T1, R2, T2)
