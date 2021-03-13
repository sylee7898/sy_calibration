''' *********************************************************** '''
''' ***************** find the vanishing point **************** '''
''' *********************************************************** '''
''' *********************************************************** '''

import numpy as np
import cv2
import glob
import os
import math
import time

import matplotlib.pyplot as plt

import itertools
import random
from itertools import starmap


''' ******************* 일단 정리******************* '''
'''
(Intrinsic Parameters)
K = [[fc,   skew*fx,    cx],
    [0,     fy,         cy], 
    [0,     0,          1]]

## p = roll, t = tilt
[Xc] =  [cos(p),    -sin(p)*sin(t),  -sin(p)*cos(t)  ],  [X-X1],
[Yc]    [sin(p),    cos(p)*sin(t),   cos(p)cos(t)    ],  [Y-Y1],
[Zc]    [0,         -cos(t),        sin(t)          ]   [Z-Z1]

#    ## p = pan, t = tilt
#    [Xc] =  [cos(p),           sin(p),        0      ],  [X-X1],
#    [Yc]    [-sin(p)*sin(t),   cos(p)*sin(t), -cos(t)],  [Y-Y1],
#    [Zc]    [-sin(p)*cos(t),   cos(p)cos(t),  sin(t) ]   [Z-Z1]

s * [x] =   K   [Xc]
    [y]         [Yc]
    [1]         [Zc]


atan2(y,x) = atan(y/x)

'''
''' *********************************************** '''
# 환경 1 : 대각에서 보는 땅 중 땅에 체커보드가 있다 >> 선 따서 사각형 검출, 체커보드화  > vertical line 구하고 horizon 계산, roll, tilt도
# (바닥에 무늬 없을 때)
# 환경 2 : 길 라인따라 vanishing_point 구할수있을 때 >> vanishing point부터 주점까지 이어 vertical_vanishing line 구하고, roll,tilt도
# 환경 2-1 : 각진 건물이나 기둥있을 때 > 바닥이 xy평면, 기둥이 z축으로 구함
# 환경 3 : 길 잘 안따질때, 키로 vanishing line (horizon line) 구할 수 있어 > horizon선이 일직선 아니고기울어야 roll, tilt도 구함
#           ex ) MOT 영상 중 ID4 키구했던 영상

# [ 바닥 체스보드화하기 ]
# 1. 바닥에 체스보드 구해 xy평면 만들기
# 바닥 체크무늬에서 vertical line 구해 (X자일때 어떤걸 vertical로 할지 : 일단은 기울기가 양수인걸로 )

# vertical_line의 slope = vy/vx 일때
# roll = atan(-a*vx/vy)    # vertical line의 수직성분(horizon_line)의 atan
# tilt = atan2(sqr(a*a*vx*vx, vy*vy), -a*f)

# R,t 구할 수 있어 >
# 파라미터로 보정!

##################
# 사람 머리랑 발 구함 >> P1 = K1[R1 t1]  ,   P2 = K2[R2 St2] 에서 S로 extrinsicParam 안구해도 되는건지 관계파악
#
#


# Perform edge detection
# img = background
# test시에는 chessBoard 넣고
def hough_transform(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale

    #start = time.time()                     # 시간 측정 시작
    kernel = np.ones((15, 15), np.uint8)

    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)  # Open (erode, then dilate)
    edges = cv2.Canny(opening, 50, 100, apertureSize=3)  # Canny edge detection     ##############100-200이 vertical_line 더 선명

    #cv2.imshow('edges', edges)
    #cv2.imwrite(output_dir + '/cannyEdges-50-150.jpg', edges)      # CannyEdge저장할거면 켜기

    minLineLength = 50     # 이 값보다 짧은 선은 찾지않음
    maxLineGap = 20         # 이 값보다 떨어져있으면 다른 직선으로 간주
    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180, threshold=100, lines=np.array([]),
                            minLineLength=minLineLength, maxLineGap=maxLineGap)


    hough_lines = []
    #print("hough time : ", time.time() - start)   # 시간 측정 종료

    numLine = img.copy()
    #line은 [x y x y] 로 들어가는데 list(lines[i][0])을 넣으면 [x, y, x, y] > line[0,0][0,1][1,0][1,1]을 1차원으로 배열한 것
    a, b, c = lines.shape       # line개수 (57), 1, 라인두점xyxy (4)
    for i in range(a):
        p1 = (lines[i][0][0], lines[i][0][1])
        p2 = (lines[i][0][2], lines[i][0][3])

        cv2.line(img, p1, p2, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.line(numLine, p1, p2, (0, 0, 255), 2, cv2.LINE_AA)
        hough_lines.append(list(lines[i][0]))       #append는 요소에 list그대로 넣고, extend는 요소하나당 list 요소 하나씩 풀어넣

        cv2.imwrite(output_dir + 'hough_(canny50,100)_'+ name + '_houghlines-20-50.jpg', img)
        if choose_line:
            cv2.putText(numLine, '%d'%i, p1, cv2.FONT_HERSHEY_PLAIN, 1, color=(0,0,0))



    if choose_line :
        #cv2.imshow('line numbers', numLine)
        if 'y' == input("저장 [y/n] ? n 추천 : ") :
            cv2.imwrite(output_dir + name + '_houghlines-20-100_num.jpg', numLine)



    return hough_lines


def endpoints(rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x_0 = a * rho
    y_0 = b * rho
    x_1 = int(x_0 + 1000 * (-b))
    y_1 = int(y_0 + 1000 * (a))
    x_2 = int(x_0 - 1000 * (-b))
    y_2 = int(y_0 - 1000 * (a))

    return ((x_1, y_1), (x_2, y_2))

# Random sampling of lines
def sample_lines(lines, size):
    if size > len(lines):
        size = len(lines)
    return random.sample(lines, size)


# vertical line 구해서 vx, vy, a, t 반환       이거 아냐 선형변환 안되고 tan모양 됨
def find_verticalline(line1, line2, CX):

    # line = [x, y, x, y]
    # a는 기울기, x는 x, m은 관계식 기울기, n은 관계식 절편
    # y = a * x + b   >>  b = y - a * x
    # cy일때 x인 xc = (cy - b) / a
    # a = m * xc + n   >>  m xc1 + n - a1  ==  m xc2 + n - a2 = 0
    # xc2*xc1*m + xc2*n - xc2*a1  ,  xc1*xc2*m + xc1*n - xc1*a2
    # (xc2 - xc1)n (-xc2*a1 + xc1*a2) = 0   >> n = (xc2*a1 - xc1*a2) / (xc2 - xc1)
    # (xc2 - xc1)m + a1 - a2 = 0   >>  m = (a2 - a1) / (xc2 - xc1)

    # a = m * (cy - xc1) + n    으로 a 구하고
    # cy = a * cx + t    >>   t 구해서 vertical 식 구하기

    # 기울기 구하는 법 영상이랑 그래프는 달라 : 영상은 x축으로 뒤집은거라 y에 -1 곱해


    if line1[1] > line1[3] :
        # 체커보드는 라인 밑에서부터
        print("ddddd")
        a1 = -(line1[3] - line1[1]) / (line1[2] - line1[0])
        a2 = -(line2[3] - line2[1]) / (line2[2] - line2[0])
    else :
        # 점을 위에서부터 찍었을 경우
        a1 = -(line1[1] - line1[3]) / (line1[0] - line1[2])
        a2 = -(line2[1] - line2[3]) / (line2[0] - line2[2])

    print("라인1의 기울기 a1 : ", a1, "\n라인2의 기울기 a2 : ", a2)

    th1 = math.atan(a1)
    th2 = math.atan(a2)


    b1 = line1[1] - a1 * line1[0]
    b2 = line2[1] - a2 * line2[0]

    # cy일 때 x점
    xc1 = (cy - b1) / a1
    xc2 = (cy - b2) / a2

    diff = xc2 - xc1
    th_diff = (th2 - th1) / diff


    '''
    n = (xc2*a1 - xc1*a2) / (xc2 - xc1)
    m = (a2 - a1) / (xc2 - xc1)
    
    a = m * (CX - xc1)
    '''
    # y = a*x + t
    m = th_diff * (CX - xc1)
    a = math.tan(m)
    t = cy - a * CX


    print("vertical line :  y = ", a, " * x + ", t)

    # vertical_line의 cy보다 +-30인 지점에서 x,y 구해서 v0에 추가
    # vx1 = (cy+30 - t) / a
    # vx2 = (cy-30 - t) / a


    # vx, vy는 두 점사이 x,y 거리차이
    vx1 = (cy + 30 - t) / a
    vx = vx1 - CX
    vy = 30

    return vx, vy, a, t

def det(a, b):
    return a[0] * b[1] - a[1] * b[0]

# Find intersection point of two lines (not segments!)
def line_intersection(line1, line2):
    x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    div = det(x_diff, y_diff)  # 행렬식 = 0 이면 기울기 같아

    if div == 0:
        return None  # Lines don't cross

    d = (det(*line1), det(*line2))
    x = det(d, x_diff) / div
    y = det(d, y_diff) / div
    if x > 0 and y > 0:  # 나중에 vanishing point 찾을 때는 음수여도 찾아야해
        return x, y


def get_crosspt(line1, line2):
    x11, y11, x12, y12 = line1
    x21, y21, x22, y22 = line2
    if x12==x11 or x22==x21:
        return None
    m1 = -(y12 - y11) / (x12 - x11)
    m2 = -(y22 - y21) / (x22 - x21)
    print("m1 : ", m1, "m2 : ", m2)
    if m1==m2:
        print('parallel')
        return None
    #print(x11,y11, x12, y12, x21, y21, x22, y22, m1, m2)
    cx = (x11 * m1 - y11 - x21 * m2 + y21) / (m1 - m2)
    cy = m1 * (cx - x11) + y11


    return cx, cy


# Find intersections between multiple lines (not line segments!)
def find_intersections(lines):
    intersections = []
    for i, line_1 in enumerate(lines):
        # print("line1 : ", line_1)
        line_1 = [line_1[0], line_1[1]], [line_1[2], line_1[3]]
        # print("line1 : ", line_1)
        for line_2 in lines[i + 1:]:
            line_2 = [line_2[0], line_2[1]], [line_2[2], line_2[3]]
            if not line_1 == line_2:
                #intersection = line_intersection(line_1, line_2)
                intersection = get_crosspt(line_1, line_2)

                if intersection:  # If lines cross, then add
                    intersections.append(intersection)

    return intersections






''' ************************************************** '''


chessX = 6              #6, 11
chessX = chessX -1
chessY = 5              #5, 8
chessY = chessY -1


##################################################################################
#############    intrinsic parameters     ########################################
##################################################################################
'''
image_height = img.shape[0]
image_width = img.shape[1]

cx = image_width / 2                                                     ####### 수정해야함
cy = image_height / 2

a = 1       # (auto_Cali 논문에선 NTSC(720*480)에선 0.91, PAL(720*576)에선 1.09)
fx = cx / tan(CameraFOV / 2)        # FOV in the horizontal direction
fy = a*fx                           # FOV in the vertical direction 넣고 fx랑 똑같이 구하던지
skew = 0

K = np.array([[fx,  skew,   cx  ],
              [0,   fy,     cy  ],
              [0,   0,      1   ]])


'''



''' *********************************************************** '''
'''
if 'y' == input("choose the line [y/n] : "):
    choose_line = True
else:
    choose_line = False
'''
choose_line = False


lines = []
intersections = []
grid_size = 60

''' ****************** PATH ********************************************************* '''
path = os.getcwd()
#image_dir = os.path.join(path, "data", "Set_1", "sample")
#output_dir = os.path.join(image_dir, "result")
#image_dir = os.path.join(path, "GML_CameraCalibrationToolboxImages", "Images")
#output_dir = os.path.join(image_dir, "result")

image_dir = '/home/seungyeon/Desktop/git/neurvps/data/line/'
output_dir = image_dir + 'rst/'

# MOT16-02-067
# MOT16-04-001
# MOT16-09-001
# MOT16-10-218
name = 'MOT16-10-218'
######## glob.glob한 변수에 + 'str' 안됨
#img = cv2.imread('./data/Set_1/sample/vertical_camera1_3-23.jpg')
img = cv2.imread(image_dir+name+'.jpg')


''' ********************************************************************************** '''
h,  w = img.shape[:2]        # 576, 704
cx = w / 2
cy = h / 2
#cv2.imshow('img', img)

vertMat = img.copy()
twover = img.copy()

start = time.time()
# vanishing point 구하기
lines = hough_transform(img)
print("hough time : ", time.time() - start)
#lines = [line][xyxy]

# lsd cv_lib : neurvps/LSD/pylsd/example_cv2.py




if choose_line:


    '''
    # 건물이나 기둥 있을 때 vertical, horizon 선택
    vertical = int(input("vertical line number : "))        # yellow
    horizon = int(input("horizon line number : "))          # Blue
    #img 수정가능한 복사
    verhor = orgin.copy()
    cv2.line(verhor, (lines[vertical][0][0], lines[vertical][0][1]), (lines[vertical][0][2], lines[vertical][0][3]), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(verhor, (lines[horizon][0][0], lines[horizon][0][1]), (lines[horizon][0][2], lines[horizon][0][3]), (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('verhor_line', verhor)
    cv2.imwrite(output_dir + '/camera1_houghlines-20-200_line.jpg', verhor)
    '''

    # 두 수직선 골라 중간 vertical line 구하기
    vertical1 = int(input("vertical1 line number : "))  # 왼쪽
    vertical2 = int(input("vertical2 line number : "))  # 오른쪽

    # img 복사

    cv2.line(img, (lines[vertical1][0], lines[vertical1][1]), (lines[vertical1][2], lines[vertical1][3]), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(img, (lines[vertical2][0], lines[vertical2][1]), (lines[vertical2][2], lines[vertical2][3]), (0, 255, 255), 2, cv2.LINE_AA)
    #cv2.imshow('twover_line', img)
    #cv2.imwrite(output_dir + '/camera1_houghlines-20-100_twover.jpg', img)


    # vanishing point 구하기
    vanPx, vanPy = get_crosspt(lines[vertical1], lines[vertical2])

    vanishing = twover.copy()
    # cx,cy말고 다른 점들도 vanishing poin랑 이어주기
    '''
    #horizon 세로 
    for i in range(-4, 5) :
        # 9개의 점 vanishing point랑 이을거야
        CX = cx + i * grid_size
        slop = -(cy - vanPy) / (CX - vanPx)
        t = cy - slop * CX
        # y = 0일때 x = -t / slop
        x1 = int(-t / slop)     # y = 0인 지점 x1
        if i == 0 :
            xx = x1

        # y = h 일때 x = (h - t) / slop
        x2 = int((h - t) / slop)

        cv2.line(vanishing, (x1, 0), (x2, h), (255, 0, 0), 2, cv2.LINE_AA)      #vertical
    '''
    #horizon 가로
    for i in range(-4, 5):
        # 9개의 점 vanishing point랑 이을거야
        CY = cy + i * grid_size
        slop = -(CY - vanPy) / (cx - vanPx)
        t = CY - slop * cx
        # y = 0일때 x = -t / slop
        x1 = int(-t / slop)  # y = 0인 지점 x1
        if i == 0:
            xx = x1

        # y = h 일때 x = (h - t) / slop
        x2 = int((h - t) / slop)

        cv2.line(vanishing, (x1, 0), (x2, h), (0, 0, 255), 2, cv2.LINE_AA)  # horizon

    # 서있는 구조물이 vertical vanishing line 만들어
    # 바닥평면이 horizon >>  h = vx*x + (vy/a)*y + f^2



    #cv2.imshow('vanishing', vanishing)
    cv2.imwrite(output_dir + '/camera2_vanishingLine.jpg', vanishing)
    '''
    vx = vanPx - cx
    vy = -(cy -vanPy)


    # roll, tilt 구해서 R, t 구하기                   #### intrinsic prameter 필요
    a = 1
    roll = math.atan2(-a * vx, vy)
    f = float(input("(6-5 3.2 // 11-8 3.6 )f : "))              # 6-5 f=3.2 / 11-8 f=3.6
    tilt = math.atan2(np.sqrt((a*a*vx*vx) + vy*vy), -a * f)     # vx,vy 길이달라지면 tilt 달라져

    print("roll : ", roll, "tilt : ", tilt)
    '''
    '''
    ## p = roll, t = tilt
    [Xc] =  [cos(p),    -sin(p)*sin(t),  -sin(p)*cos(t)  ],  [X-X1],
    [Yc]    [sin(p),    cos(p)*sin(t),   cos(p)cos(t)    ],  [Y-Y1],
    [Zc]    [0,         -cos(t),         sin(t)          ]   [Z-Z1]

    '''
    '''
    #다커 식이 맞음
    R = [[math.cos(roll),   -math.sin(roll)*math.sin(tilt), -math.sin(roll)*math.cos(tilt)],
         [math.sin(roll),   math.cos(roll)*math.sin(tilt),  math.cos(roll)*math.cos(tilt)],
         [0,                -math.cos(tilt),                math.sin(tilt)              ]]
    #이게 논문 식인데
#    R = [[math.cos(roll),   -math.sin(roll) * math.cos(tilt),   -math.sin(roll) * -math.sin(tilt)],
#        [math.sin(roll),    math.cos(roll) * math.cos(tilt),    math.cos(roll) * -math.sin(tilt)],
#        [0,                 math.sin(tilt),                     math.cos(tilt)                  ]]

    print(R)
    '''



cv2.waitKey()
