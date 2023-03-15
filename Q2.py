import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math
import sympy as sym
import scipy as sp
from scipy.misc import derivative
from sympy import *
from itertools import groupby, product
import imutils
import random

sift = cv.SIFT_create()
def homography(points):
    A=[]
    Homography_matrix=[]
    for i in range(0,points.shape[0]):
        a_1=[points[i][0],points[i][1],1,0,0,0,-points[i][2]*points[i][0],-points[i][2]*points[i][1],-points[i][2]]
        a_2=[0,0,0,points[i][0],points[i][1],1,-points[i][3]*points[i][0],-points[i][3]*points[i][1],-points[i][3]]
        A.append(a_1)
        A.append(a_2)
    #A.pop(0)
    A=np.array(A)
    A_t=A.transpose()
    A_t_A=np.matmul(A_t,A)
    eig_val,eig_vec = np.linalg.eig(A_t_A)
    for i in range(9):
        if(min(eig_val)==eig_val[i]):
            Homography_matrix=eig_vec[:,i]
    Homography_matrix=np.reshape(Homography_matrix,(3,3))
    Homography_matrix=Homography_matrix/Homography_matrix[2][2]
    return Homography_matrix

def crop(img):
    gray_image=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    threshold=cv.threshold(gray_image,0,255,cv.THRESH_BINARY)[1]
    mask=np.transpose(np.nonzero(threshold))
    x_min=0
    y_min=0
    y_max=0
    x_max=0
    for k in range(mask.shape[0]):
        if mask[k][0]==0 and mask[k][1]>y_max:
            y_max=mask[k][1]
        if mask[k][1]==y_max and mask[k][0]>x_max:
            x_max=mask[k][0]
    img=img[y_min:y_max,x_min:x_max]
    return img

def distance(pts,H):
    print(pts)
    p_1=np.array([pts[0],pts[1],1])
    p_1=p_1.T
    p_2_approx=np.dot(H,p_1)
    p_2_approx=(1/p_2_approx[2])*p_2_approx
    p_2=np.array([pts[0],pts[1],1]).T
    error=p_2-p_2_approx
    return np.linalg.norm(error)

def ransac(pts,threshold):
    max_inliers=[]
    H_final=None
    for i in range(500):
        p1=pts[random.randrange(0,len(pts))]
        p2=pts[random.randrange(0,len(pts))]
        pts_homography=np.vstack((p1,p2))
        p3=pts[random.randrange(0,len(pts))]
        pts_homography=np.vstack((pts_homography,p3))
        p4=pts[random.randrange(0,len(pts))]
        pts_homography=np.vstack((pts_homography,p4))
    H=homography(pts_homography)
    inliers=[]
    for i in range(len(pts)):
        d=distance(pts[i],H)
        if d<threshold:
            inliers.append(pts[i])
    if len(inliers)>len(max_inliers):
        max_inliers=inliers
        H_fin=H
    return H_fin 

def getSiftKpDes(image1, image2, image3, image4):

    ## Detect features from images
    keypoints_1, descriptors_1 = sift.detectAndCompute(image1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image2,None)
    keypoints_3, descriptors_3 = sift.detectAndCompute(image3,None)
    keypoints_4, descriptors_4 = sift.detectAndCompute(image4,None)

    return keypoints_1,keypoints_2,keypoints_3,keypoints_4, descriptors_1, descriptors_2, descriptors_3, descriptors_4

def getPoints(kp1, kp2, dsc1, dsc2):
    points=[]
    #bf=cv.BFMatcher()
    #bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
    bf = cv.BFMatcher()

    #matches=bf.match(dsc1,dsc2)
    matches=bf.knnMatch(dsc1,dsc2,k=2)
    
    #matches = sorted(matches, key = lambda x:x.distance)
    #best_match = matches[0:50]

    best_match = []
    best_match_list = []
    for m,n in matches:
        if m.distance < (0.7*n.distance) :
            best_match.append(m)
    for val in best_match:
        (x_img1,y_img1)=kp1[val.queryIdx].pt 
        (x_img2,y_img2)=kp2[val.trainIdx].pt
        points.append([int(x_img1),int(y_img1),int(x_img2),int(y_img2)])
    points=np.array(points)
    return points, best_match

img1 = cv.imread('image_1.jpg')
img2 = cv.imread('image_2.jpg')
img3 = cv.imread('image_3.jpg')
img4 = cv.imread('image_4.jpg')

img1 = cv.resize(img1, (640,480), interpolation=cv.INTER_AREA)
#img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

img2 = cv.resize(img2, (640,480), interpolation=cv.INTER_AREA)
#img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

img3 = cv.resize(img3, (640,480), interpolation=cv.INTER_AREA)
img4 = cv.resize(img4, (640,480), interpolation=cv.INTER_AREA)

#Obtain Keypoint and descriptors using the below getSiftKpDes function
kp1, kp2, kp3, kp4, dsc1, dsc2, dsc3, dsc4 = getSiftKpDes(img1, img2, img3, img4)

pts_12, best_match_12 = getPoints(kp1, kp2, dsc1, dsc2)
H_12=homography(pts_12)
matched_12 = cv.drawMatches(img1, kp1, img2, kp2, best_match_12, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#warped_12=cv.warpPerspective(img2,np.linalg.inv(H_12),(img1.shape[1]+img2.shape[1],img1.shape[0]+img2.shape[0]))
warped_12=cv.warpPerspective(img2,np.linalg.inv(H_12),(img1.shape[1]+img2.shape[1],img1.shape[0]+img2.shape[0]))
warped_12[0:img1.shape[0], 0:img1.shape[1]] = img1

pts_34, best_match_34 = getPoints(kp3, kp4, dsc3, dsc4)
H_34=homography(pts_34)

#warped_12=cv.warpPerspective(img2,np.linalg.inv(H_12),(img1.shape[1]+img2.shape[1],img1.shape[0]+img2.shape[0]))
warped_34=cv.warpPerspective(img4,np.linalg.inv(H_34),(img3.shape[1]+img4.shape[1],img3.shape[0]+img4.shape[0]))
warped_34[0:img3.shape[0], 0:img3.shape[1]] = img3

warped_12 = crop(warped_12)

warped_34 = crop(warped_34)

#sift = cv.SIFT_create()
kp12, dsc12 = sift.detectAndCompute(warped_12,None)
kp34, dsc34 = sift.detectAndCompute(warped_34,None)

pts_123, best_match_123 = getPoints(kp12, kp3, dsc12, dsc3)
H_123=homography(pts_123)
warped_123=cv.warpPerspective(img3,np.linalg.inv(H_123),(int((warped_12.shape[1]+img3.shape[1])*0.75),int((warped_12.shape[0]+img3.shape[0])*0.7)))
warped_123[0:warped_12.shape[0], 0:warped_12.shape[1]] = warped_12

warped_123 = crop(warped_123)
kp123, dsc123 = sift.detectAndCompute(warped_123,None)
pts_1234, best_match_1234 = getPoints(kp123, kp4, dsc123, dsc4)
H_1234=homography(pts_1234)
warped_1234=cv.warpPerspective(img4,np.linalg.inv(H_1234),(int(warped_123.shape[1]+(img4.shape[1]*0.2)),int(warped_123.shape[0]+(img4.shape[0]*0.2))))
warped_1234[0:warped_123.shape[0], 0:warped_123.shape[1]] = warped_123
warped_1234 = crop(warped_1234)

cv.imshow("Warped images 1 and 2", warped_12)
cv.imshow("warped images 1,2 and 3", warped_123)
cv.imshow("warped all images", warped_1234) 
cv.imshow("Best match", matched_12)

#cv.imshow("warped images 3 and 4", warped_34)


cv.waitKey(0)
cv.destroyAllWindows()
