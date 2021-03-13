#########################################################
#########################################################
######        intrinsic Parameters          #############
#########################################################
#########################################################

from __future__ import print_function, division
import os
import glob
import sys, argparse
import pprint
from typing import List, Any

import numpy as np
import cv2
from scipy import optimize as opt
import math

# from skspatial.objects import Point, Line
import deep_sort_another as gta


np.set_printoptions(suppress=True)
puts = pprint.pprint

#DATA_DIR = "./GML_CameraCalibrationToolboxImages/Images/"
#DEBUG_DIR = "./GML_CameraCalibrationToolboxImages/Images/result/"

image = cv2.imread('/home/seungyeon/Desktop/git/7_camera/seq_0/vpdata/1.jpeg')
rootdir = '/home/seungyeon/Desktop/git/7_camera/seq_1/vpdata/'
#name = '5'
Cam = (input("cam_num : "))
name = Cam
###############################
#######################################
'''
# 기존엔 xxx, yyy, zzz
with np.load(rootdir + name +".npz") as npz:
    vpts = np.array([npz[d] for d in ["x", "y", "z"]])
    print(vpts)

'''

# GTA 축별 vp :  xyz, xyz, xyz
vpts = []
with np.load(rootdir + name +".npz") as npz:
    # vpts_pd = np.array(npz["vpts_pd"])
    vpts_pd = np.array([npz[d] for d in ["x", "y", "z"]])
    # print("vp : ", vpts_pd)   ######################################################


x = np.zeros(len(vpts_pd))
y = np.zeros(len(vpts_pd))
for i in range(len(vpts_pd)):

    vp = np.array([vpts_pd[i][0], vpts_pd[i][1], vpts_pd[i][2]])
    # vp /= LA.norm(vp)
    vpts.append(vp)
    x[i] = vp[0] / vp[2] * 256 + 256    # x/256 -1 했었음
    y[i] = -vp[1] / vp[2] * 256 + 256   # 1 - y/256 했었음

    # 512에서 원래사이즈로
    x[i] *= image.shape[1] / 512
    y[i] *= image.shape[0] / 512

    # print(x[i], y[i])   ######################################################


##### new f 찾기 ###########################
#### f구하기 :  vp & calib 논문 ############################################################
'''
x1x2 + y1y2 + ff = 0
x2x3 + y2y3 + ff = 0
x3x1 + y3y1 + ff = 0
'''
# npz에 저장된 vpt는 world 좌표
# 이미지 2D좌표로 변환하려면 f*256+cx 곱해줘야해 >> f 알아야해
'''
x1 = vpts[0][0] / vpts[0][2]#  * f * 256 + cx
y1 = vpts[0][1] / vpts[0][2]#  * f * 256 + cy
x2 = vpts[1][0] / vpts[1][2]#  * f * 256 + cx
y2 = vpts[1][1] / vpts[1][2]#  * f * 256 + cy
x3 = vpts[2][0] / vpts[2][2]#  * f * 256 + cx
y3 = vpts[2][1] / vpts[2][2]#  * f * 256 + cy

print(x1, y1)
print(x2, y2)
print(x3, y3)
'''
x1 = x[0]
x2 = x[1]
x3 = x[2]
y1 = y[0]
y2 = y[1]
y3 = y[2]

# theta = (math.atan(y1/(x1+0.001)), math.atan(y2/(x2+0.001)), math.atan(y3/(x3+0.001)))
# p = (math.sqrt(x1**2+y1**2), math.sqrt(x2**2+y2**2), math.sqrt(x3**2+y3**2))


'''
x1 sin(theta[0]) - y1 cos(theta[0]) = 0
x2
x3
-----------------
p1 = n1 f   //  p2 = n2 f   //  p3 = n3 f
'''


# ncos1 = -(math.cos(theta[1] - theta[2])) / (math.cos(theta[0] - theta[1])) * math.cos(theta[2] - theta[0])
# ncos2 = -(math.cos(theta[2] - theta[1])) / (math.cos(theta[0] - theta[1])) * math.cos(theta[1] - theta[2])
# ncos3 = -(math.cos(theta[0] - theta[1])) / (math.cos(theta[1] - theta[2])) * math.cos(theta[2] - theta[0])
# if ncos1 < 0 :
#     ncos1 = -ncos1
# if ncos2 < 0 :
#     ncos2 = -ncos2
# if ncos3 < 0 :
#     ncos3 = -ncos3
# n1 = math.sqrt(ncos1)
# n2 = math.sqrt(ncos2)
# n3 = math.sqrt(ncos3)
#
# f1 = p[0]/n1
# f2 = p[1]/n2
# f3 = p[2]/n3
# print(f1, f2, f3)
# f = (f1+f2+f3)/3
# #f = f1
# print("f = ", f)
'''
x1 = p[0] cos(theta[0]) ,   y1 = p[0] sin(theta[0])
x2 = p[1] cos(theta[1]) ,   y2 = p[1] sin(theta[1])
x2 = p[2] cos(theta[2]) ,   y3 = p[2] sin(theta[2])
'''

####################################   f 다시 구하기    ###################################################
#################################### f 는 norm(oc.oi)=sqrt[(oc.vi)**2-(oi.vi)**2]    ####################
cx = image.shape[1]
cy = image.shape[0]

m = -(x2-x1) / (y2-y1)  # 기울기 역수가 projection
b = cy - (m*cx)
m0 = (y2-y1) / (x2-x1)
b0 = y1 - (m0*x1)

x = (b0 - b) / (m - m0)
y = m0*x + b0
y1 = m*x + b

def ddiff (x1, y1, x2, y2) :
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)



#oi = np.array([cx, cy])
#vi = np.array([x,y])
# ocvi = np.sqrt(np.dot(np.linalg.norm(ddiff(x1, y1, x, y)), np.linalg.norm(ddiff(x, y, x2, y2))))
ocvi = np.sqrt(ddiff(x1, y1, x, y) * ddiff(x, y, x2, y2))
f = np.sqrt(abs((np.linalg.norm(ocvi))**2 - (np.linalg.norm(ddiff(cx, cy, x, y)))**2))
# print("f : ", f)  ######################################################
f = 2.4

K = [[f, 0, cx], [0, f, cy], [0, 0, 1]]








#########################################################
#########################################################
######        extrinsic Parameters          #############
#########################################################
#########################################################
# roll, tilt로 구한 R !!!!!!!!!!!!!!!!!!!

'''
p1 = (vpts[0][0]/vpts[0][2], vpts[0][1]/vpts[0][2])
p2 = (vpts[1][0]/vpts[1][2], vpts[1][1]/vpts[1][2])
p3 = (vpts[2][0]/vpts[2][2], vpts[2][1]/vpts[2][2])

roll_down = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
roll_up = abs(p1[1]-p2[1])
roll = math.asin(roll_up/roll_down)
if (p1[0]-p2[0]) / (p1[1]-p2[1]+0.001) < 0 :
    roll = -roll
print("roll : ", roll)
# 영상 원점 말고 vp Z축의 2D 점으로 구함
#tilt_d = np.cross(p2-p1, p3-p1)/np.linalg.norm(p2-p1)
tilt_d = abs((p2[0]-p1[0])*(p1[1]-p3[1]) - (p1[0]-p3[0])*(p2[1]-p1[1])) / np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
tilt = math.atan(tilt_d/f)
'''

#*f*256+cx  *f*256+cy
p1 = (x1, y1)
p2 = (x2, y2)
p3 = (x3, y3)


roll = math.atan2(-x3, y3)
tilt = math.atan2(np.sqrt(x3**2+y3**2), -f)

R = [[math.cos(roll), -math.sin(roll)*math.cos(tilt), math.sin(roll)*math.sin(tilt)],
     [math.sin(roll), math.cos(roll)*math.cos(tilt), math.cos(roll)*-math.sin(tilt)],
     [0, math.sin(tilt), math.cos(tilt)]]

# print("pre R (roll,tilt): \n", R1)
#
#
# OFu = np.array([x1-cx, y1-cy, f])        # OF 자체를 잘못구함
# OFv = np.array([x2-cx, y2-cy, f])
# print(np.dot(OFu, OFv))
# s1 = np.linalg.norm(OFu)
# s2 = np.linalg.norm(OFv)
# urc = OFu / np.linalg.norm(OFu)
# vrc = OFv / np.linalg.norm(OFv)
# urc2 = np.transpose(np.array([x1/s1, y1/s1, f/s1]))
# vrc2 = np.transpose(np.array([x2/s2, y2/s2, f/s2]))
#
#
# w = np.cross(urc2, vrc2)
#
# R = [[x1/(np.sqrt(x1**2+y**2+f**2)), x2/(np.sqrt(x2**2+y2**2+f**2)), w[0]],
#      [y1/(np.sqrt(x1**2+y**2+f**2)), y2/(np.sqrt(x2**2+y2**2+f**2)), w[1]],
#      [f/(np.sqrt(x1**2+y**2+f**2)), f/(np.sqrt(x2**2+y2**2+f**2)), w[2]]]
# print("new R : \n", R)
#
# print(urc, urc2)
# print(vrc, vrc2)



# print("R-est : ", R[0], "\n", R[1], "\n", R[2])   ######################################################

# t = 1 0 0 | 0
#     0 1 0 | 0
#     0 0 1 | -hc

# t = np.array([[0, 0, 0], [0, 0, 0], [0, 0, -50]])
'''
t = np.array([0, 0, -0.5])
'''
if Cam == '1':
    t = np.array([24.1702, 356.413, 116.933])
    r = np.array([-28.086, 0, 57.358])
elif Cam == '2':
    t = np.array([-249.48, 413.402, 113.312])
    r = np.array([-22.59, 0, -86.17])
elif Cam == '3':
    t = np.array([-313.33, 448.293, 112.291])
    r = np.array([-30.72, 0, -105.827])
elif Cam == '4':
    t = np.array([-99.9358, 415.335, 117.92])
    r = np.array([-25.95, 0, 65.325])
elif Cam == '5':
    t = np.array([-350.424, 382.829, 114.374])
    r = np.array([-29.2322, 0, -84.299])
elif Cam == '6':
    t = np.array([-484.801, 404.051, 103.784])
    r = np.array([-18.67, 0, -123.753])
else:
    t = np.array([-89.691, 288.912, 110.26])
    r = np.array([-20.44, 0, 43.269])


Rt = np.hstack((R, t.reshape(-1, 1)))
# Rt[0][3] = 0
# Rt[1][3] = 0
# Rt[2][3] = -t[2]
print("Rt : \n", Rt)


KRt = np.dot(K, Rt)     # 3*4 Mat
KR = np.dot(K, R)



# print("KR : ", KR)    ######################################################

############# v1, v2, v3 = KR ###########
# KR = np.array(vpts)
# KR[0] = KR[0]/np.linalg.norm(KR[0])

################################################## R 새로 구하기 ##############################
'''
rvecs = [-36.5879, -53.1652e-07, -101.79]
tvecs = [-393.237, 520.018, 125.619]

# 역 파라미터 구하
K = np.array(K).reshape(3, 3).astype(np.float32)
rvecs = np.array(rvecs).reshape(1, 3).astype(np.float32)
tvecs = np.array(tvecs).reshape(3, 1).astype(np.float32)

# Rt-1 . ( K-1 . uv)
R, _ = cv2.Rodrigues(rvecs)
print(R)
# z축 없으니까 R3지우고 t로 넣어 Rt 3*3으로 만들기
R[0][2] = tvecs[0]
R[1][2] = tvecs[1]
R[2][2] = tvecs[2]
print("R : ", R)

KR = np.dot(K, R)
'''

KR_inv = np.linalg.inv(KR)
KRt_inv = np.linalg.pinv(KRt)       # 4*3 Mat




####################################################################################################
'''    test    '''
# vz는 h의 vp로 vpts 마지막꺼
# vl은 t,b 중간지점?
# t vl b vz 있을 때     (vl, vz는 일정하고 t,b만 변함)
#  (t-b)(vz-vl) / (t-vl)(vz-b) = 알파Z        ## Z는 실제 사람 크기

# 알파Z = (-|b x t|) / (I^T b |vz x t|)               ################################################################

########### uv상의 두길이 비교
obj1 = (1146, 67, 1141, 157)        # 870 이미지 왼위, 오른아래
obj2 = (1636, 218, 1581, 366)
# obj1 =  (1343, 114, 1314, 240)
# obj2 =  (1487, 56, 1468, 141)

'''
t1 = np.array(obj1[0:2])
b1 = np.array(obj1[2:])
vl1 = np.array((t1-b1)/2)
vz = np.array(p3)

t2 = np.array(obj2[0:2])
b2 = np.array(obj2[2:])
vl2 = np.array((t2-b2)/2)

diff1 = (t1-b1)*(vz-vl1) / (t1-vl1)*(vz-b1)
diff2 = (t2-b2)*(vz-vl2) / (t2-vl2)*(vz-b2)
print(diff1)
diff11 = np.sqrt(diff1[0]**2 + diff1[1]**2)
diff22 = np.sqrt(diff2[0]**2 + diff2[1]**2)
print("diff1 : ", diff11)
print("diff2 : ", diff22)
'''

# obj는 물제 길이 (x1, y1, x2, y2)
def diff (obj1, obj2, KR_inv) :
    # 두 경첩의 4포인트를 XYZ로 변환
    Wobj1_t = np.dot(KR_inv, np.transpose([obj1[0], obj1[1], 1]))
    Wobj1_b = np.dot(KR_inv, np.transpose([obj1[2], obj1[3], 1]))
    Wobj2_t = np.dot(KR_inv, np.transpose([obj2[0], obj2[1], 1]))
    Wobj2_b = np.dot(KR_inv, np.transpose([obj2[2], obj2[3], 1]))
    # print(Wobj1_t, Wobj1_b, Wobj2_t, Wobj2_b)

    # XYZ 길이 파악
    # world = [0, 1, 2, 3]
    Wobj1 = (Wobj1_t[0] - Wobj1_b[0], Wobj1_t[1] - Wobj1_b[1], Wobj1_t[2] - Wobj1_b[2], Wobj1_t[3] - Wobj1_b[3])  # 위아래 포인트 X,Y,Z 차이값
    Wobj2 = (Wobj2_t[0] - Wobj2_b[0], Wobj2_t[1] - Wobj2_b[1], Wobj2_t[2] - Wobj2_b[2], Wobj2_t[3] - Wobj2_b[3])

    # DIFF1 = np.sqrt((Wobj1[0] / Wobj1[2]) ** 2 + (Wobj1[1] / Wobj1[2]) ** 2)  # 얘가 맞음
    # DIFF2 = np.sqrt((Wobj2[0] / Wobj2[2]) ** 2 + (Wobj2[1] / Wobj2[2]) ** 2)
    DIFF1 = np.sqrt((Wobj1[0]) ** 2 + (Wobj1[1]) ** 2 + (Wobj1[2])**2)  # 얘가 맞음
    DIFF2 = np.sqrt((Wobj2[0]) ** 2 + (Wobj2[1]) ** 2 + (Wobj2[2])**2)

    print("DIFF1 : ", DIFF1)  # X/Z, Y/Z
    print("DIFF2 : ", DIFF2)

    print("scale : ", (172/DIFF1))
    print("DIFF1 = 172 >> DIFF2 : ", DIFF2*(172/DIFF1))



def test_world (KRt_inv):
    gt_dir = '/home/seungyeon/Desktop/git/7_camera/seq_1/' + str(Cam)
    gt = open(gt_dir + '/gt.txt', 'r')
    wd = open(gt_dir + '/world.txt', 'r')

    # for l, w in gt, wd:
    #     frame = int(l.split(',')[0])
    #
    #     bbox[0] = int(l.split(',')[2])
    #     bbox[1] = int(l.split(',')[3])
    #     bbox[2] = int(l.split(',')[4])
    #     bbox[3] = int(l.split(',')[5])
    #
    #     XYZ = np.array([w.split(',')[2:5]])
    #     # print(XYZ)
    #
    #     # gt로 뽑은 좌표는 uv
    #     # P * XY = xy
    #     uv = np.array([bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2])
    #     xy = np.dot(KRt, XYZ)
    #
    #     # uv랑 xy 비교


    ################
    # 거리비교

    '''
    실험 1 : cam1_6 : 20-60
    pre = np.array([1016,34,24,60])
    cur = np.array([995,11,22,49])
    xy_pre = np.array([-12.220, 382.660, 112.450])
    xy_cur = np.array([-20.500,387.410,112.600])
    
    실험 2 : cam2_1 : 20-100
    pre = np.array([1009,186,41,81])
    cur = np.array([823,127,19,52])
    xy_pre = np.array([-220.080,413.640,109.480])
    xy_cur = np.array([-201.090,421.730,109.830])
    
    실험 3 : cam3_4 : 20-100
    pre = np.array([953,75,53,125])
    cur = np.array([1038,-28,27,70])
    xy_pre = np.array([-293.280,442.210,108.060])
    xy_cur = np.array([-275.000,434.340,108.080])
    
    실험 4 : cam4_2 : 100-150
    pre = np.array([-128,364,117,166])
    cur = np.array([982,333,88,151])
    xy_pre = np.array([-116.410,409.610,113.040])
    xy_cur = np.array([-112.090,421.820,113.080])
    
    실험 5 : cam5_2 : 910-1010
    pre = np.array([810,-38,16,42])
    cur = np.array([930,0,23,63])
    xy_pre = np.array([-283.470,397.060,110.240])
    xy_cur = np.array([-307.160,387.810,110.200])
    
    실험 6 : cam6_7 : 10-100
    pre = np.array([137,49,27,56])
    cur = np.array([127,33,22,42])
    xy_pre = np.array([-430.000,403.290,106.300])
    xy_cur = np.array([-408.470,403.370,108.350])
    
    실험 7 : cam7_6 : 900-1000
    pre = np.array([861,78,17,44])
    cur = np.array([899,123,26,67])
    xy_pre = np.array([-135.170,330.670,110.690])
    xy_cur = np.array([-117.000,315.720,108.870])
    
    
    
    
    추가1 : cam3_3 : 350-400
    pre = np.array([1390,250,97,192])
    cur = np.array([1079,47,45,110])
    xy_pre = np.array([-303.420,440.140,107.790])
    xy_cur = np.array([-290.850,438.970,108.080])
    
    추가2 : cam6_5 : 10-110
    pre = np.array([1373,189,47,106])
    cur = np.array([1710,144,34,69])
    xy_pre = np.array([-470.560,384.580,102.170])
    xy_cur = np.array([-467.370,364.810,103.000])
    '''

    # ID = int(input("ID : "))
    # start = int(input("start frame : "))
    # end = int(input("end frame : "))

    # gt[start].split(',')[2]
    # bbox 중앙 위치

    # pre = np.array([ gt[start].split(',')[2] + gt[start].split(',')[4]/2, gt[start].split(',')[3] + gt[start].split(',')[5]/2 ])
    # cur = np.array([ gt[end].split(',')[2] + gt[end].split(',')[4]/2, gt[end].split(',')[3] + gt[end].split(',')[5]/2 ])

    if(int(Cam)==1):
        pre = np.array([1016, 34, 24, 60])
        cur = np.array([995, 11, 22, 49])
        xy_pre = np.array([-12.220, 382.660, 112.450])
        xy_cur = np.array([-20.500, 387.410, 112.600])

    elif(int(Cam)==2):
        pre = np.array([1009, 186, 41, 81])
        cur = np.array([823, 127, 19, 52])
        xy_pre = np.array([-220.080, 413.640, 109.480])
        xy_cur = np.array([-201.090, 421.730, 109.830])

    elif(int(Cam)==3):
        pre = np.array([953, 75, 53, 125])
        cur = np.array([1038, -28, 27, 70])
        xy_pre = np.array([-293.280, 442.210, 108.060])
        xy_cur = np.array([-275.000, 434.340, 108.080])

        # pre = np.array([1390, 250, 97, 192])
        # cur = np.array([1079, 47, 45, 110])
        # xy_pre = np.array([-303.420, 440.140, 107.790])
        # xy_cur = np.array([-290.850, 438.970, 108.080])

    elif(int(Cam)==4):
        pre = np.array([-128, 364, 117, 166])
        cur = np.array([982, 333, 88, 151])
        xy_pre = np.array([-116.410, 409.610, 113.040])
        xy_cur = np.array([-112.090, 421.820, 113.080])

    elif(int(Cam)==5):
        pre = np.array([810, -38, 16, 42])
        cur = np.array([930, 0, 23, 63])
        xy_pre = np.array([-283.470, 397.060, 110.240])
        xy_cur = np.array([-307.160, 387.810, 110.200])

    elif(int(Cam)==6):
        pre = np.array([137, 49, 27, 56])
        cur = np.array([127, 33, 22, 42])
        xy_pre = np.array([-430.000, 403.290, 106.300])
        xy_cur = np.array([-408.470, 403.370, 108.350])

        # pre = np.array([1373, 189, 47, 106])
        # cur = np.array([1710, 144, 34, 69])
        # xy_pre = np.array([-470.560, 384.580, 102.170])
        # xy_cur = np.array([-467.370, 364.810, 103.000])

    else:
        pre = np.array([861, 78, 17, 44])
        cur = np.array([899, 123, 26, 67])
        xy_pre = np.array([-135.170, 330.670, 110.690])
        xy_cur = np.array([-117.000, 315.720, 108.870])




    pre0 = (pre[0] + pre[2]/2, pre[1] + pre[3]/2)
    cur0 = (cur[0] + cur[2]/2, cur[1] + cur[3]/2)
    uv_pre = np.dot(KRt_inv, np.transpose([pre0[0], pre0[1], 1]))
    uv_cur = np.dot(KRt_inv, np.transpose([cur0[0], cur0[1], 1]))

    uv_diff = np.sqrt((uv_pre[0]-uv_cur[0])**2 + (uv_pre[1]-uv_cur[1])**2 + (uv_pre[2]-uv_cur[2])**2 + (uv_pre[3]-uv_cur[3])**2)



    h_c = np.array([cur[0], cur[1], cur[0], cur[1] + cur[3]])
    h_t = np.dot(KRt_inv, np.transpose([h_c[0], h_c[1], 1]))
    h_b = np.dot(KRt_inv, np.transpose([h_c[2], h_c[3], 1]))
    h_w = (h_t[0] - h_b[0], h_t[1] - h_b[1], h_t[2] - h_b[2], h_t[3] - h_b[3])
    h_diff = np.sqrt((h_w[0]) ** 2 + (h_w[1]) ** 2 + (h_w[2]) ** 2)
    scale = 1.72 / h_diff
    print(scale)


    wuv_diff = scale * uv_diff

    # world x,y,z위치
    # XY_pre = np.array([ wd[start].split(',')[2] + wd[start].split(',')[4]/2, wd[start].split(',')[3] + wd[start].split(',')[5]/2 ])
    # XY_cur = np.array([ wd[end].split(',')[2] + wd[end].split(',')[4]/2, wd[end].split(',')[3] + wd[end].split(',')[5]/2 ])
    xy_diff = np.sqrt((xy_pre[0]-xy_cur[0])**2 + (xy_pre[1]-xy_cur[1])**2 + (xy_pre[2]-xy_cur[2])**2)

    print("uv_diff [m] : ", uv_diff)
    print("xy_diff [m] : ", xy_diff)




    gt.close()
    wd.close()


# diff(obj1, obj2, KRt_inv)         # 870경첩 0.8981, 0.8995
test_world(KRt_inv)