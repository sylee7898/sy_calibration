# https://github.com/nwojke/deep_sort


'''
det 정보없이 tracking gt에서 bbox만 가지고 (tracking안하고) ID별 속도 계산     후 거리 계산 

[ Calibratino에 쓰는 전처리용 트레킹 ]
키 계산하기
이전 카메라의 방정식 불러와
이번 카메라의 값 두세개정도 비교해 S 구하기       > calibration으로 값 주고
///////////////////////
calibration에서 파라미터 받아 > velocity 계산 
한 영상에서 ID의 속도 구하기
다른영상에서 나타나는 시간 * ID속도 >> 카메라간 거리 구하기
'''


# https://github.com/nwojke/deep_sort
#####################################################################################
##### 얘는 속도만 찾는걸로 [h구하는 부분 다 지우고 deepsort another에 h랑 S구하는거 모아 ] #####
#####################################################################################
'''
python deep_sort_app.py \
    --sequence_dir=./data/MOT16/train/MOT16-02 \
    --detection_file=./resources/detections/MOT16_POI_train/MOT16-02.npy \
    --output_file=./result/text/MOT16-02.txt \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --display=True
'''

# vim: expandtab:ts=4:sw=4
#from __future__ import division, print_function, absolute_import

import argparse
import os
from typing import List, Any, Union

import cv2
import numpy as np
# import matplotlib.pyplot as plt

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

# import velocity as vel_py
import self_parameters as param


# framerate는 15정도로 설정
# tracking 다 지우고 txt로 id별 bbox추출
# foot 추출 해서 거리재기 (world로)
# frame으로 시간 계산해서 ID별 속력 저장
# ID가 사라진 시간 * 저장된 속력으로 zone 거리측정




# ID 고르고
# cam 미리 설정해줘서 해당 gt.txt 들어감
# 설정하는 frame 위치중에서 ID와 같은 부분 bbox로 거리 계산

def run(velocity):


    gt_dir = '/home/seungyeon/Desktop/git/7_camera/seq_1/' + str(Cam)
    bbox = np.zeros(4)

    with open(gt_dir + '/gt.txt', 'r') as gt:
        count = 1
        foot_diff = []
        sum_diff = 0

        # endT = (range_end - range_start) * update_ms
        # print("설정한 end time : ", endT)



        for l in gt:
            frame = int(l.split(',')[0])
            if (frame >= range_start and frame <= range_end):
                ID = int(l.split(',')[1])
                if (ID == IDnum) :
                    bbox[0] = int(l.split(',')[2])
                    bbox[1] = int(l.split(',')[3])
                    bbox[2] = int(l.split(',')[4])
                    bbox[3] = int(l.split(',')[5])


                    cur_foot = (bbox[0] + (bbox[2] / 2), bbox[1] + bbox[3], 1)    # [x,y,z,1]이어야하는데 z는 0이라 R에서 r3를 뺏으니 여기도 빼
                    if count == 1 :
                        # 첫프레임이면 foot, diff 초기화
                        cur_foot_w = XYworld(cur_foot, KR_inv)
                        pre_foot_w = cur_foot_w
                        # pre_foot = cur_foot
                        # sum_diff = 0
                        count = 2
                        sum_h = height(bbox)
                        count_bbox = 1


                    # 초기화 후에 반복하여 계산
                    # elif count > 1 and frame % 5 == 1:
                    elif count > 1:
                        # ID 처음 나타나는 프레임부터   5 frame마다 차이거리 재기

                        # if pre_foot == cur_foot:
                        #     print("발위치 그대로")  ################ 프린트 부분 한번도 안들어감 발위치 계속 바뀌나봐
                        #     diff = 0
                        #     if count == 1:
                        #         # 처음 사라진 시간 저장
                        #         print("사라진 count = 1")
                        #         nolookT = frame * update_ms
                        #         count = 2
                        #     elif count > 1:
                        #         print("사라진 count = ", count)
                        #         count += 1
                        #         if count == 11 or frame > (range_end - 1):
                        #             # 20frame씩 열번 돌동안 안나타나거나 안나타난채로 마지막 프레임이면 속도 반환하고 끝내는데
                        #             # 시간은 처음 사라진 시간에 사라졌다 생각하고 그시간동안만 속도 구하기
                        #             velocity = sum_diff / nolookT
                        #             print("%d 에 추정마침.\n속도 : %f" % (nolookT, velocity))
                        #             count = -1

                        # else:  # 발위치 변하면 계산
                            # print("20frame 후 발위치 변함")
                        count += 1  # 사라졌었는데 다시 나타나면 count = 1로 다시 사라지는 때 기다림
                        cur_foot_w = XYworld(cur_foot, KR_inv)
                        diff = diff_xy(pre_foot_w, cur_foot_w)
                        sum_diff += diff



                        count_bbox += 1
                        sum_h += height(bbox)  # bbox 크기 부정확해서 들어오는 모든 bbox  world 길이로 평균내서 저장
                        # h = sum_h / count_bbox
                        # print("%f,     %f" % (height(bbox), h))

                        if frame > (int(range_end) - 5):  # 정해놓은 시간만큼 돌아가서 마지막 프레임 오면 끝내

                            time = (frame - range_start) * update_ms
                            velocity = sum_diff / time
                            print("%s camera" % Cam)
                            # print("총 거리 : ", sum_diff)
                            print("%d ms 동안 추적.  속도 : %f" % (time, velocity))
                            count = 0

                        foot_diff.append(diff)  # 발 위치 변화량 보려고
                        pre_foot_w = cur_foot_w
                        # pre_foot = cur_foot

        h = sum_h / count_bbox
        scale = 172 / h
        print("scale : ", scale)
        velocity = scale * velocity
        print("절대 속력 [cm/ms] : ", velocity)

        return velocity





def XYworld(cur_foot, KR_inv):

    #foot 열이 2개이면 3개로 늘려주기
    if len(cur_foot) == 2 :
        cur_foot = np.array(cur_foot[0], cur_foot[1], 1)


    #XY = np.dot(R_inv, np.dot(K_inv, cur_foot))
    XY = np.dot(KR_inv, cur_foot)
    XYw = np.array([XY[0], XY[1]])

    #print("XY : ", XY)
    #print("XYw : ", XYw)

    return XYw

def diff_xy(pre_foot, cur_foot):
    # print("pre_foot : ", pre_foot)
    # print("cur_foot : ", cur_foot)

    # 발 위치는 z=0이므로 z값은 제외하고 계산
    Xw = pre_foot[0] - cur_foot[0]
    Yw = pre_foot[1] - cur_foot[1]

    diff = np.sqrt(Xw * Xw + Yw * Yw)
    # print("diff : ", diff)

    return diff

def height(bbox):

    Wobj_t = np.dot(KR_inv, np.transpose([bbox[0], bbox[1], 1]))
    Wobj_b = np.dot(KR_inv, np.transpose([bbox[0], bbox[1] + bbox[3], 1]))

    # print(Wobj_t)
    # print(Wobj_b)
    # XYZ 길이 파악
    Wobj_t3 = np.array([Wobj_t[0], Wobj_t[1], Wobj_t[2]])
    Wobj_b3 = np.array([Wobj_b[0], Wobj_b[1], Wobj_b[2]])

    Wobj = (Wobj_t3[0] - Wobj_b3[0], Wobj_t3[1] - Wobj_b3[1], Wobj_t3[2] - Wobj_b3[2])  # 위아래 포인트 X,Y,Z 차이값
    DIFF = np.sqrt((Wobj[0] ** 2) + (Wobj[1] ** 2) + (Wobj[2] ** 2))
    # print("ID bbox로 측정한 키 (스케일 전) : ", DIFF)
    return DIFF





if __name__ == "__main__":


    IDnum = int(input("ID_num : "))
    # Cam = int(input("cam_num for tracking : "))
    Cam = param.Cam
    range_start = int(input("range start : "))  #########
    range_end = int(input("range end : "))  #########

    ######################################
    frameRate = 10
    update_ms = 1000 / int(frameRate)
    velocity = 0

    # KR_inv = param.KR_inv

    KRt_inv = param.KRt_inv
    KR_inv = KRt_inv            ###################### 코드 안바꾸기위해 KR_inv로 변수명 통일

    veloc = run(velocity)

    # 거리 = 시간 * 속력
    t1 = range_end
    print("사라진 frame : ", t1)
    t2 = int(input("나타난 frame : "))
    T = (t2-t1) * update_ms
    distance = T * veloc

    print("카메라간 거리 [cm] = ", distance)






