import numpy as np
import cv2 as cv
#import scipy.signal as sig
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import math
import sympy as sym
import scipy 
from scipy.misc import derivative
from sympy import *
from itertools import groupby, product
from scipy.spatial.transform import Rotation as Ro

toRad = math.pi/180
pts = ()
phi = []
theta_val =[]
psi = []
x_coordinate = []
y_coordinate = []
z_coordinate = []
numofFrames = 0
def computeHomography(corner_points):
    """ x_w = np.array([0,0,27.9,27.9])
    y_w = np.array([0, 21.6, 0, 21.6]) """
    l=27.9
    w=21.6
    x_w = np.array([0,0,l,l])
    y_w = np.array([0,w,0,w])
    A = np.array((8,9),dtype=float)
    A = []
    """ x_cam =[]
    y_cam =[] """
    j=0
    for points in corner_points:
        x,y = points
        A.append([x,y,1,0,0,0,-x_w[j]*x,-x_w[j]*y,-x_w[j]])
        A.append([0,0,0,int(x),int(y),1,-y_w[j]*x,-y_w[j]*y,-y_w[j]])
        j+=1
    #print(A.shape)
    #print(A)
    A = np.array(A)
    A_t_A = np.matmul(A.transpose(),A)
    #print(A_t_A)
    eig_val, eig_vec = np.linalg.eig(A_t_A)
    min_eig = min(eig_val)
    
    for i in range(0,8):
        if min_eig == eig_val[i]:
            h = eig_vec[:,i]
    
    return h

def calculateCameraPose(grouped_pts,H):
    k = np.array([[1382.58398, 0, 945.743164], [0, 1383.57251, 527.04834],[0, 0, 1]])
    A = np.matmul(np.linalg.pinv(k),H)
    print("k-inv and H multiplied is:")
    print(A)
    lambda_val = 0
    """ for j in range(0,3):
        lambda_val += np.square(A[j][0])
    lambda_val = np.sqrt(lambda_val) """
    a1, a2 = np.linalg.norm(A[:,0]), np.linalg.norm(A[:,1])
    lambda_val = (a1+a2)/2
    print("Lambda value is")
    print(lambda_val)

    r1 = np.zeros(3,dtype=float)
    r2 = np.zeros(3,dtype=float)
    r1 = np.hstack([A[0][0]/lambda_val, A[1][0]/lambda_val, A[2][0]/lambda_val])
    r2 = np.hstack([A[0][1]/lambda_val, A[1][1]/lambda_val, A[2][1]/lambda_val])
    t = np.hstack([A[0][2]/lambda_val, A[1][2]/lambda_val, A[2][2]/lambda_val])
    r3 = np.cross(r1,r2)
    
    #Rot_trans_mat = np.vstack([r1,r2,r3,t])
    Rot_trans_mat = []
    Rot_trans_mat.append(r1)
    Rot_trans_mat.append(r2)
    Rot_trans_mat.append(r3)
    Rot_trans_mat.append(t)
    Rot_trans_mat = np.array(Rot_trans_mat)
    Rot_trans_mat = Rot_trans_mat.T
    print("The rotational translational matrix is:")
    print(Rot_trans_mat)

    pose = np.matmul(k,Rot_trans_mat)
    print("The camera pose is:")
    print(pose)
    print(pose.shape)
    return r1,r2,r3,t


def calculate_xy(dval, angles, idx1, idx2):
    A = np.array([[math.cos(angles[idx1]*toRad), math.sin(angles[idx1]*toRad)],[math.cos(angles[idx2]*toRad), math.sin(angles[idx2]*toRad)]])
    d= np.array([[dval[idx1]], [dval[idx2]]])
    mat = np.matmul(np.linalg.inv(A),d)
    pt = (int(mat[0]),int(mat[1]))
    corner_pts.append(pt)
    return corner_pts

def Manhattan_dist(tup1, tup2):
        return abs(tup1[0] - tup2[0]) + abs(tup1[1] - tup2[1])

def grouping(corner_pts):
    man_tups = [sorted(sub) for sub in product(corner_pts, repeat = 2) if Manhattan_dist(*sub) <= 20]
    
    res_dict = {ele: {ele} for ele in corner_pts}
    for tup1, tup2 in man_tups:
        res_dict[tup1] |= res_dict[tup2]
        res_dict[tup2] = res_dict[tup1]
    
    res = [[*next(val)] for key, val in groupby(sorted(res_dict.values(), key = id), id)]
    
    mean = []
    for val in res: 
        mean_val = np.mean(val, axis=0)
        mean.append(mean_val)
    mean_sorted = sorted(mean, key=lambda x: x[0])
    return mean_sorted


# Open Video file
cap = cv.VideoCapture('project2.avi')
#cap = cv.VideoCapture('project2.avi')

# Check if the video opened successfully
if (cap.isOpened()== False): 
  print("The file cannot be opened")

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()         
  if ret == True:
    flag = 0
    image=frame.copy()
    imgGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    x=[]
    y=[]
    corner_pts = []
    gaussian_blur = cv.GaussianBlur(imgGray,(15,15),0)
    edges = cv.Canny(gaussian_blur,75,150)
    y, x = np.where(edges != 0)

    H = {}
    
    for i in range(0,len(x)):
        for theta in range(0,180):
                d =  x[i]*math.cos(theta*toRad) + y[i]*math.sin(theta*toRad)
                key = (np.round(d),theta)
                if key not in H:
                    H[key] = 1
                else:
                    H[key] += 1
    
   
    #print(H)
    #sortedKeys = sorted(H)
    key_list = []
    sorted_dict = sorted(H.items(), key=lambda x:x[1])[-20:]
    for i in sorted_dict:
        key_list.append(i[0])
    #print(key_list)
    
    d_val = [] # hold the d values
    angles = [] # hold the angles
    for a,angle in key_list:
        d_val.append(a)
        angles.append(angle)
    
    #print(angles)
    corners = []
        
    for index1 in range(0,len(angles)):
        for index2 in range(index1+1,len(angles)):
            if abs(angles[index1] - angles[index2]) >= 85 and abs(angles[index1] - angles[index2]) <= 95:
                corners= calculate_xy(d_val, angles, index1, index2)

        else:
            pass
    
    grouped_corners = grouping(corners)
    print("the number of points are:")
    print(len(grouped_corners))
    for pts in grouped_corners:
        numofFrames = numofFrames+1
        cv.circle(frame, (int(pts[0]),int(pts[1])), 5, (0,0,255), -1)

    homography_mat = computeHomography(grouped_corners)
    print(homography_mat)
    for i in range(0,9):
        homography_mat[i] = homography_mat[i]/homography_mat[8]
    
    homography_sqmat = np.zeros((3,3), dtype=float)
    k=0
    for i in range(0,3):
        for j in range(0,3):
            homography_sqmat[i][j] = homography_mat[k]
            k+=1
    print(homography_sqmat)

    r1,r2,r3,t = calculateCameraPose(grouped_corners,homography_sqmat)
    R = np.zeros((3,3), dtype=float)
    print(r1)
    print(r2)
    print(r3)
    R[:,0] = r1
    R[:,1] = r2
    R[:,2] = r3


    print("The Rotation Matrix is:") 
    print(R)
    
    r = Ro.from_matrix(R)
    phi.append(r.as_euler('zyx', degrees=True)[0])
    theta_val.append(r.as_euler('zyx', degrees=True)[1])
    psi.append(r.as_euler('zyx', degrees=True)[2])

    x_coordinate.append(float(t[0]))
    y_coordinate.append(float(t[1]))
    z_coordinate.append(float(t[2]))


    cv.imshow("Edges", edges)
    cv.imshow("Normal image with center", frame) 

    # Press Q on keyboard to  exit
    if cv.waitKey(10) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything is done, release the video capture object
cap.release()
 
# Closes all the frames
cv.destroyAllWindows()

### Plotting data
fig=plt.figure()
ax=fig.add_subplot(221,projection="3d")
ax.set_title ("Translation matrix plotted")
ax.scatter3D(x_coordinate,y_coordinate,z_coordinate, c= "green", label='Translation')
plot_frames = numofFrames//4
x_plot_val = np.linspace(1,plot_frames,num=plot_frames)
ax=fig.add_subplot(222)
ax.set_title ("x rotation angle")
plt.plot(x_plot_val, phi, color = "blue", label = "x rotation angle")

ax=fig.add_subplot(223)
ax.set_title ("y rotation angle")
plt.plot(x_plot_val, theta_val, color = "blue", label = "y rotation angle")

ax=fig.add_subplot(224)
ax.set_title ("z rotation angle")
plt.plot(x_plot_val, psi, color = "blue", label = "z rotation angle")
plt.show()
