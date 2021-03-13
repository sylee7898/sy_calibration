# calibration, deep_sort_another 다 전처리고
# 얘가 실시간

############
# f, R, t 필요

# 일정 속도의 발 점 받아오기
# 이전 발 위치와 현재 발 위치 카메라 좌표로 변환
# K-1 * uv로 구하기
#  k =  [  f   0   cx],
#       [  0   f   cy],
#       [  0   0    1]




############ 필요없는 import 지우기
from __future__ import division, print_function, absolute_import

import argparse
import os
from typing import List, Any, Union

import cv2
import numpy as np
#import matplotlib.pyplot as plt

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

#import deep_sort_app as deep        # 테스트용으로 여기서 run돌리
import self_parameters

# class Velocity(nn.Module) :
#
#     def __init__(self, KR_inv):
#         self.KR_inv = KR_inv

KR_inv = self_parameters.KR_inv

path = os.getcwd()
# param_dir = path + '/data/Set_1/sample/result/'         # click_camera3_param.txt
# video_dir = path + '/data/Set_1/ID_03/Camera_3/Seq_1/'      # 맨오른쪽 검은남자 세로,가로 다 움직
# param_dir = path + '/GML_CameraCalibrationToolboxImages/Images/result/'
# video_dir = path + '/GML_CameraCalibrationToolboxImages/Images/'
video_dir = path + '/data/MOT16/train/MOT16-02/img1/'

# img = cv2.imread(video_dir + 'video0000.jpg')
img = cv2.imread(video_dir + 'PICT0001.JPG')


#### GTA gt camera param
'''
def param(self):

    h, w = img.shape[:2]
    cx = w / 2
    cy = h / 2

    #with open(param_dir + 'click_camera3_param.txt', 'r') as f_param:
    with open(param_dir + 'chessboard_param_6-5.txt', 'r') as f_param:
        fx = float(f_param.read().splitlines()[0].split(' ')[0])
        # fy = float(f_param.read().splitlines()[0].split(' ')[1])
        # focal = (fx + fy) / 2       ######### click으로 f 잘 안나와서 둘 차이 크면 사이값으로 / 이럴거면 아무숫자나;;
        focal = fx
        print(focal)
    f_param.close()


    K = [[focal, 0, cx],
         [0, focal, cy],
         [0, 0, 1]]
    print(K)


    # 얘도 f_param txt에서 추출해야해
    #rvecs = [-0.12370166, 0.24478434, 1.92741803]           ## R,t 파라미터 구하기
    #tvecs = [-2.28651134, -3.79107266, 42.09720921]
    rvecs = [-36.5879, -53.1652e-07, -101.79]
    tvecs = [-393.237, 520.018, 125.619]

    ##################################################
    #역 파라미터 구하
    K = np.array(K).reshape(3, 3).astype(np.float32)
    rvecs = np.array(rvecs).reshape(1, 3).astype(np.float32)
    tvecs = np.array(tvecs).reshape(3, 1).astype(np.float32)

    # Rt-1 . ( K-1 . uv)
    R, _ = cv2.Rodrigues(rvecs)
    
    
    # z축 없으니까 R3지우고 t로 넣어 Rt 3*3으로 만들기
    R[0][2] = tvecs[0]
    R[1][2] = tvecs[1]
    R[2][2] = tvecs[2]
    print("R : ", R)


    #R_inv = np.linalg.inv(R)
    #K_inv = np.linalg.inv(K)

    KR = np.dot(K, R)
    KRinv = np.linalg.inv(KR)

    return KRinv   #K_inv, R_inv
'''

# self_parameter랑 연동해서 가져오기
'''
def param_vp() :
    # KRt = vp [xxx][yyy][zzz]

    #MOT16-02
    KR= [[3.19311696e+04,  1.07043270e+03, - 1.12686642e+02],
        [-9.31087772e+02,  2.63588193e+04, - 1.80480979e+04],
        [0.00000000e+00,   5.89504017e-01,   8.07765445e-01]]
    
    #MOT16-09
    KR = [[ 8.09356474e+02,  1.91417246e+04, -2.49746151e+03],
         [-1.92964073e+04,  9.50092027e+02,  8.20964702e+02],
         [ 0.00000000e+00,  1.57063399e-01,  9.87588522e-01]]

    KR_inv = np.linalg.inv(KR)

    return KR_inv
'''

# 실시간 발자국 변환이니까 실시간으로 각자 변환만하고
# 이전 발위치는 이미 계산되어있으니 차이값만 구하기
def XYworld(cur_foot):

    #foot 열이 2개이면 3개로 늘려주기
    if len(cur_foot) == 2 :
        cur_foot = np.array(cur_foot[0], cur_foot[1], 1)


    #XY = np.dot(R_inv, np.dot(K_inv, cur_foot))
    XY = np.dot(KR_inv, cur_foot)
    XYw = np.array([XY[0] / XY[2], XY[1] / XY[2], 1])

    #print("XY : ", XY)
    #print("XYw : ", XYw)

    return XYw

def diff_xy(pre_foot, cur_foot):
    print("pre_foot : ", pre_foot)
    print("cur_foot : ", cur_foot)

    Xw = pre_foot[0] - cur_foot[0]
    Yw = pre_foot[1] - cur_foot[1]

    diff = np.sqrt(Xw * Xw + Yw * Yw)
    print("diff : ", diff)

    return diff



def foot_world(frame_idx, track_id, bbox, IDnum, update_ms, max_frame_idx, endT) :
    global count, foot_diff, first_frame, endFrame, pre_foot, cur_foot, pre_foot_w, cur_foot_w, diff, sum_diff, nolookT, velocity, K_inv, R_inv
    if IDnum == 0 :
        return 0
    # bbox = track.to_tlwh()
    # ID가 처음 발견될 때 초기화
    # frame 번호 저장하고 그에 따른 endTime 설정하기
    if int(track_id) == int(IDnum) and count == 0:
        first_frame = frame_idx
        count = 1

        # 첫프레임이면 foot, diff 초기화
        cur_foot = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3], 1)
        cur_foot_w = XYworld(cur_foot, KR_inv)
        pre_foot_w = cur_foot_w
        pre_foot = cur_foot
        # sum_diff = 0

        # frameRate는 초당 몇frame인지 > 1장당 걸린시간은 1초/frameRate
        # seq_info["update_ms"] 가 프레임당 몇ms 걸리는지
        # 시간은 frame * update_ms
        # endT = 10  # 반복 종료할 시간 [초]는 인자로 넣어주기
        endFrame = (endT * 1000 / update_ms) + first_frame
        if endFrame > max_frame_idx:
            endFrame = max_frame_idx
            endT = (endFrame - first_frame) * update_ms
        print("설정한 end time : ", endT)
        print("설정한 end Frame : ", endFrame)
        print("First Frame : ", first_frame)
        print("10frame마다 발위치 측정")

    # 초기화 후에 반복하여 계산
    if count > 0 and frame_idx % 10 == 1:
        # ID 처음 나타나는 프레임부터   20 frame마다 차이거리 재기
        #if int(track_id) == int(IDnum):
        #    print("ID detect")
        cur_foot = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3], 1)    # [x,y,z,1]이어야하는데 z는 0이라 R에서 r3를 뺏으니 여기도 빼

        if pre_foot == cur_foot:
            print("발위치 그대로")            ################ 프린트 부분 한번도 안들어감 발위치 계속 바뀌나봐
            diff = 0
            if count == 1:
                # 처음 사라진 시간 저장
                print("처음 사라짐 count = 1")
                nolookT = frame_idx * update_ms
                count = 2
            elif count > 1:
                print("사라진 count = ", count)
                count += 1
                if count == 11 or frame_idx > (endFrame - 1):
                    # 20frame씩 열번 돌동안 안나타나거나 안나타난채로 마지막 프레임이면 속도 반환하고 끝내는데
                    # 시간은 처음 사라진 시간에 사라졌다 생각하고 그시간동안만 속도 구하기
                    velocity = sum_diff / nolookT
                    print("%d 에 추정마침.\n속도 : %f" % (nolookT, velocity))
                    count = -1

        else:  # 발위치 변하면 계산
            #print("20frame 후 발위치 변함")
            count = 1  # 사라졌었는데 다시 나타나면 count = 1로 다시 사라지는 때 기다림
            cur_foot_w = XYworld(cur_foot, KR_inv)
            diff = diff_xy(pre_foot_w, cur_foot_w)
            sum_diff += diff

            if frame_idx > (int(endFrame) - 1):  # 정해놓은 시간만큼 돌아가서 마지막 프레임 오면 끝내
                velocity = sum_diff / endT
                print("%d 에 추정마침.\n속도 : %f" % (endT, velocity))
                count = -1

        foot_diff.append(diff)  # 발 위치 변화량 보려고
        pre_foot_w = cur_foot_w
        pre_foot = cur_foot








######################################
# foot_world 하기위한 초기화
count = 0
foot_diff = []
first_frame = 0
endFrame = 0
pre_foot = 0
cur_foot = 0
pre_foot_w = 0
cur_foot_w = 0
diff = 0
sum_diff = 0
nolookT = 0
velocity = 0

# K_inv, R_inv = param()
# KR_inv = param_vp()

######################################################################
# 영상좌표를 동일하게 움직였을 때 월드좌표는 x,y축 다르게 계산되는거 확인
# 체커보드 에서 네모 한칸을 월드로 변환하면 1:1로 나타남
'''
uv1 = [858.79626, 536.0956, 1]
uv2 = [830.41846, 604.07404, 1]  # 아래점

uv3 = [789.2872 , 511.2483, 1]  # 왼쪽점


K_inv, R_inv = param()

XY1 = np.dot(R_inv, np.dot(K_inv, uv1))
XYw1 = np.array([XY1[0] / XY1[2], XY1[1] / XY1[2], 1])
XY2 = np.dot(R_inv, np.dot(K_inv, uv2))
XYw2 = np.array([XY2[0] / XY2[2], XY2[1] / XY2[2], 1])
XY3 = np.dot(R_inv, np.dot(K_inv, uv3))
XYw3 = np.array([XY3[0] / XY3[2], XY3[1] / XY3[2], 1])
print("XY1 : \n", XY1)
print("XYw1 : \n", XYw1)
print("XY2 : \n", XY2)
print("XYw2 : \n", XYw2)
print("XY3 : \n", XY3)
print("XYw3 : \n", XYw3)
Xw12 = XYw1[0] - XYw2[0]
Yw12 = XYw1[1] - XYw2[1]
Xw13 = XYw1[0] - XYw3[0]
Yw13 = XYw1[1] - XYw3[1]
print(np.sqrt(Xw12 * Xw12 + Yw12 * Yw12))  # xy1 - 2
print("*****************")
print(np.sqrt(Xw13 * Xw13 + Yw13 * Yw13))  # xy1 - 3

# world좌표 diff 값은 둘다 1에 가깝게 나왔음 > 실제 크기에 맞게 스케일링만 해주면 실제값 측정가능 
'''

#########################################################################



