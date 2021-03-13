import numpy as np
import cv2
import glob
import os

import math

##################################################
#######   체커보드 파람 or 클릭   ###################
##################################################

########################################
# self_parameters로 intrinsic 구해오기
# calibration_self 로 vanishing point 구해 평면라인 구하고 / tilt,roll 구하기로 extrinsic
# 요기에서 extrinsic으로 click 변환

# 요기에서는 체커보드로 parameter 구해주거나
# 체커보드 없을때 어떻게할지 self_parameters, calibration_self 이용해서 구하

''' ******************* 일단 정리******************* '''
'''
(Intrinsic Parameters)
K = [[fc,   skew*fx,    cx],
    [0,     fy,         cy], 
    [0,     0,          1]]
## p = pan, t = tilt                                                    # p roll인지 pan인지 구해보
[Xc] =  [cos(p),           sin(p),        0      ],  [X-X1],
[Yc]    [-sin(p)*sin(t),   cos(p)*sin(t), -cos(t)],  [Y-Y1],
[Zc]    [-sin(p)*cos(t),   cos(p)cos(t),  sin(t) ]   [Z-Z1]

s * [x] =   K   [Xc]
    [y]         [Yc]
    [1]         [Zc]


atan2(y,x) = atan(y/x)

'''
''' *********************************************** '''


chessX = 6              #6, 11
chessX = chessX -1
chessY = 5              #5, 8
chessY = chessY -1
chessSize = chessX * chessY


# termination criteria      반복연산 종료 기준
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,     # type / ITER, EPS, or both
            30,         # max number of iterations
            0.001)      # min accuracy

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)                    ######################
#objp = np.zeros((chessY * chessX, 3), np.float32)
#objp[:, :2] = np.mgrid[0:chessX, 0:chessY].T.reshape(-1, 2)     # reshape 2차원으로 바꿔주는데 -1은 알아서 해주는거라 뒤에 2 넣었으니 행은 전체 요소수 / 2개로 자동 채워짐
                                                                # mgrid[0:a, 0:b] > Broadcasting을 위한 차원 부풀림 : a줄, b개씩 array 늘림 >> [1,2,...,b], [2,...], ..., [a,...]

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

imgPtxt = []    # 2d points 이미지별로 보기용 > txt로 저장

''' ***************** PATH ******************************************************* '''
path = os.getcwd()
images = glob.glob(path + '/GML_CameraCalibrationToolboxImages/Images/*.JPG')
#images = glob.glob(path + '/data/Set_1/sample/*.jpg')
#images = glob.glob(path + '/GML_CameraCalibrationToolboxImages/calibration_wide/*.jpg')
#images = glob.glob(path + '/GML_CameraCalibrationToolboxImages/Images/PICT0001.JPG')
#images = glob.glob(path + '/data/Set_1/sample/vanishingLine_camera3.jpg')


result_dir = path + '/GML_CameraCalibrationToolboxImages/Images/result/'
#result_dir = path + '/data/Set_1/sample/result/'
#result_dir = path + '/GML_CameraCalibrationToolboxImages/calibration_wide/result/'
''' ****************************************************************************** '''

#############################
pressing = False
click_corner = []
click_x = 0
click_y = 0


def mouse_callback(event, x, y, flags, param):
    global click_corner, click_x, click_y, imgNum
    if(event == cv2.EVENT_LBUTTONDOWN) :
        '''
        click_x = x
        click_y = y
        
        # corner 점이랑 가까이 있으면 그 점으로 바꾸기
        for a in range(chessSize):
            if (int(corners[a+chessSize*imgNum][0]) > click_x - 10 and int(corners[a+chessSize*imgNum][0]) < click_x + 10):
                if (int(corners[a+chessSize*imgNum][1]) > click_y - 10 and int(corners[a+chessSize*imgNum][1]) < click_y + 10):
                    ax = corners[a+chessSize*imgNum][0]
                    ay = cprmers[a+chessSize*imgNum][1]
                    break
            else:
                ax = x
                ay = y
        print("(%d, %d) >> (%d, %d)" % (x, y, ax, ay))
        '''
        
        click_corner.append((x, y))
        cv2.circle(img, (x, y), 2, (255, 0, 0), -1)



if 'y' == input("click point [y/n] : ") :
    pressing = True
    chessX = int(input("chessX : "))
    chessY = chessX
    chessSize = chessX * chessY
else : pressing = False



objp = np.zeros((chessY * chessX, 3), np.float32)
objp[:, :2] = np.mgrid[0:chessX, 0:chessY].T.reshape(-1, 2)

imgNum = 0
for fname in images:

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    imgPtxt.append(fname)       # 이미지별 포인트 보기 용
    imgPtxt.append('\n')


    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (chessX, chessY), None)
    # If found, add object points, image points (after refining them)
    if ret == True :

        objpoints.append(objp)

        # 코너 발견하면 정확도 높이기
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        imgPtxt.append(corners2)


        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (chessX, chessY), corners2, ret)

        #cv2.imshow('img', img)
        #cv2.imwrite(fname+'_corner 11-8.JPG', img)
        cv2.waitKey(500)


    elif(pressing == True) : # 체커보드 아닌경우 코너점 검출해서 마우스로 선택
        # find Harris corners
        pressing = True
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)    # 코너 검출하고
        dst = cv2.dilate(dst, None)
        rett, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
        dst = np.uint8(dst)

        # find centroids
        rett, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

        # define the criteria to stop and refine the corners
        #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001) # 이미 설정함
        corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

        for i in range(len(corners)) :
            cv2.circle(img, tuple(corners[i]), 2, (0,0,255), -1)

        print("좌상단부터 상단>하단 순으로 클릭!")
        cv2.namedWindow('img')
        cv2.setMouseCallback('img', mouse_callback)


        while(1) :
            cv2.imshow('img', img)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break



        imgpoints.append(click_corner)
        imgPtxt.append(click_corner)

        objpoints.append(objp)

        imgNum += 1



        #### 일단 코너점 잡히는거 잘 잡히는지 보고
        #### 마우스 이용해서 코너점 선택
        #### 선택한 사각형 모서리를 이용해 uv > XY로 변환


objpoints = np.array(objpoints).reshape(-1,chessSize,3).astype(np.float32)
imgpoints = np.array(imgpoints).reshape(-1,chessSize,2).astype(np.float32)

'''
    # fname 의 칸 크기
    # u,v를 XY롤 환선해서 XY거리 실제거리로 바꾸기 
    # chessXsize = 0,1사이 거리
    # chessYsize = 0,chessX사이 거리

    if chessX == 5 :
        chessLen = 30     #30mm
    if chessX == 10 :
        chessLen = 21    #21mm
'''

# 메트릭스, 왜곡 계수, 회전/이동 벡터 등이 반환
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("mtx : \n", mtx)
print("dist : \n", dist)
print("rvecs : \n", rvecs)
print("tvecs : \n", tvecs)
##############################################################################

# Store results.
if (pressing):
    ######## 클릭 포인트일때 ##########

    uv = open(result_dir + 'click_Chess001_XYuv.txt', 'w')
    print("CLICK POINT", file=uv)
    print(imgPtxt, file=uv)
    uv.close()

    param = open(result_dir + 'click_Chess001_param.txt', 'w')
    print(mtx[0][0], mtx[1][1], file=param)     # fx,fy 먼저 저장
    print("mtx : \n", mtx, file=param)
    print("dist : \n", dist, file=param)
    print("rvecs : \n", rvecs, file=param)
    print("tvecs : \n", tvecs, file=param)

    param.close()

elif(not pressing) :
    ######## 체스보드일때 ############

    uv = open(result_dir + 'chessboard_XYuv_6-5.txt', 'w')
    print(imgPtxt, file=uv)
    uv.close()

    param = open(result_dir + 'chessboard_param_6-5.txt', 'w')
    print(mtx[0][0], mtx[1][1], file=param)     # fx,fy 먼저 저장
    print("mtx : \n", mtx, file=param)
    print("dist : \n", dist, file=param)
    print("rvecs : \n", rvecs, file=param)
    print("tvecs : \n", tvecs, file=param)

    param.close()


##########################################################################


'''
6*5
mtx : 
 [[3.19865879e+03 0.00000000e+00 1.03198711e+03]
 [0.00000000e+00 3.20903048e+03 8.24219194e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
dist : 
 [[-4.90695204e-01  7.22607453e+00 -4.45325393e-03  2.29534333e-03  -3.80484509e+01]]
rvecs : 
 [array([[ 0.39247877],       [-1.00606643],       [-2.89978369]]),
  array([[-0.25787606],       [-0.15831318],       [-0.95541075]]),
  array([[-0.14180203],       [ 0.87652996],       [ 1.77913078]]),
  array([[-0.06936633],       [ 0.01678392],       [ 3.05748929]]),
  array([[-0.05167884],       [ 0.1370202 ],       [ 1.89863097]]),
  array([[ 0.15814613],       [-0.11967936],       [-1.97384314]]),
  array([[-0.12370166],       [ 0.24478434],       [ 1.92741803]]),
  array([[-0.59202982],       [ 0.53190778],       [ 1.4210733 ]])]
tvecs : 
 [array([[ 2.68104315],       [-4.59249855],       [35.63159564]]),
  array([[ 3.26091697],       [ 2.19401566],       [32.67494197]]), 
  array([[-5.80897697],       [-3.80102338],       [39.43384432]]), 
  array([[ 3.53100626],       [-4.89403941],       [39.92737892]]), 
  array([[-3.42193052],       [-3.22646547],       [28.9242331 ]]), 
  array([[ 0.77308675],       [-1.85580426],       [40.57416029]]), 
  array([[-2.28651134],       [-3.79107266],       [42.09720921]]), 
  array([[-3.20354385],       [-2.40400753],       [31.54029082]])]

'''

# 점 잘 찾는거까지 다 잘 되면 여기 수행
# 왜곡 제거 말고 uv 점으로 XY점 찾는거
# darkprogrammer #########################################################
'''
# intrinsic Parameters & distCoeff 알아야해
cam_matrix = np.array(mtx).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(dist).reshape(5, -1).astype(np.float32)
objPoints = np.array(objpoints).reshape(8,-1,3).astype(np.float32)
imgPoints = np.array(imgpoints).reshape(8,-1,2).astype(np.float32)


# 한장만 처리    (사진마다 r,t 다름)
# img 001은 [6]
#ret, rvec_, tvec_ = cv2.solvePnP(objPoints[6], imgPoints[6], cam_matrix, dist_coeffs)


# objPoints[6], imgPoints[6], rvecs[6], tvecs[6] 사용하기

rvec = np.array(rvec_).reshape(1,3).astype(np.float32)
tvec = np.array(tvec_).reshape(1,3).astype(np.float32)

print("solvePnP rvec : \n", rvec)
print("solvePnP tvec : \n", tvec)

# Rodrigues(rvec) = [3 by 3]  &  [3 by 9]
R,_ = cv2.Rodrigues(rvec)


# uv = K R t XYZ
print("K : ", cam_matrix)
print("R : ", R)
print("t : ", tvec)
A = cam_matrix * R * tvec
print("A : ", A)

A_inv = np.linalg.inv(A)
objPoints2[6] = A_inv * imgPoints[6]
'''
# 얘도 roll,tilt,pan구하긴데 얘가 원본
'''
R_inv = np.linalg.inv(R)

Cam_pos = -R_inv * tvec
#p = (float)(Cam_pos.data)
X = Cam_pos[0]
Y = Cam_pos[1]
Z = Cam_pos[2]
print("X : ", X)
print("Y : ", Y)
print("Z : ", Z)

unit_z = [0, 0, 1]
Zc = np.array(unit_z).reshape(3,1).astype(np.float32)
#Zc(3, 1, CV_64FC1, unit_z)
Zw = R_inv * Zc
zw = Zw[2]
print("zw : ", zw)
#zw = (float)(Zw.data)

pan = math.atan2(zw[1], zw[0]) - math.pi/2
tilt = math.atan2(zw[2], math.sqrt(zw[0]*zw[0]+zw[1]*zw[1]))

unit_x = [1, 0, 0]
Xc = np.array(unit_x).reshape(3,1).astype(np.float32)
#Xc(3, 1, CV_64FC1, unit_x)
Xw = R_inv * Xc
xw = Xw[0]
print("xw : ", xw)
#xw = (float)(Xw.data)
xpan = [math.cos(pan), math.sin(pan), 0]

roll = math.acos(xw[0]*xpan[0] + xw[1]*xpan[1] + xw[2]*xpan[2])
#if(xw[2] < 0) : roll = -roll

print ("pan : ", pan)
print ("roll : ", roll)
print ("tilt : ", tilt)




R_th = [[math.cos(roll), -math.sin(roll) * math.cos(tilt), -math.sin(roll) * -math.sin(tilt)],
     [math.sin(roll), math.cos(roll) * math.cos(tilt), math.cos(roll) * -math.sin(tilt)],
     [0, math.sin(tilt), math.cos(tilt)]]
print(R_th)


#reprojectdst, _ = cv2.projectPoints(img, rvec_, tvec_, cam_matrix, dist_coeffs)
#reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))
'''


###########################################################################

# 보정 전에 cv2.getOptimalNewCameraMatrix() 함수를 이용해 먼저 카메라 메트릭스 개선
# 스케일링 인자 alpha = 0 일 경우, 원치않는 픽셀을 최소로 갖는 보정된 이미지가 얻어지지만 코너지점 픽셀들이 제거될수도
# alpha = 1일 경우 모든 픽셀은 유지
# 결과를 자르는데 사용할 수 있는 이미지 ROI를 반환

image = cv2.imread('./GML_CameraCalibrationToolboxImages/calibration_wide/GOPR0032.jpg')
#image = cv2.imread(images[0])

# 아무거나 한 카메라에 한장 가져와 intrinsic parm 으로 왜곡 보정하기
cv2.imshow('image', image)
h,  w = image.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))


# 왜곡제거
dst = cv2.undistort(image, mtx, dist, None, newcameramtx)

x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv2.imwrite(result_dir + '/calibresult.png', dst)


'''
# 에러구하기
tot_error = 0
# 이미지 8장 가져와 projectPoints 구하고 에러계산
# objpoints는 사진별로 코너점 3차원 (z는 다 0)
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    tot_error += error

print("total error: ", tot_error / len(objpoints))


cv2.destroyAllWindows()
'''

''' ************************************* '''
############## extrinsic ###############
## R - roll, tilt, pan 구하기 ###########

def extrinsic_dark (rvecs, tvecs):
    ###### img 001만 각도 구함함
    rvecs = np.array(rvecs[6]).reshape(1, 3).astype(np.float32)
    tvecs = np.array(tvecs[6]).reshape(3, 1).astype(np.float32)


    image_points = []
    object_points = []

    #correspondences = selfparam.getChessboardCorners()        # Chessboard 코너 구함

    N = len(image_points)
    # object_points (N, 코너개수, xy좌표) 3차원
    '''
    for i in range(N):
        for j in range(len(image_points[i])) :  # 코너 수만큼 x,y 계산
            X = object_points[i][j][0]  # model points
            Y = object_points[i][j][1]  # model points
            u = image_points[i][j][0]  # image points
            v = image_points[i][j][1]  # image points
    '''

    # uv  =  K  Rt  XY

    R, _ = cv2.Rodrigues(rvecs)
    R_inv = np.linalg.inv(R)
    Cam_pos = np.dot(R, tvecs)
    print("Cam_pos dot : \n", Cam_pos)
    #p = (double)(Cam_pos.data)
    #Xc = np.dot(Cam_pos, X)
    #Yc = np.dot(Cam_pos, Y)

    unit_z = [0, 0, 1]
    Zc = np.array(unit_z).reshape(3, 1).astype(np.float64)
    #Zc(3, 1, CV_64FC1, unit_z)
    zw = np.dot(R_inv, Zc)
    #zw = (double)(Zw.data)
    print("[0,0,1] >> ", Zc, zw)

    pan = math.atan2(zw[1], zw[0]) - math.pi / 2
    tilt = math.atan2(zw[2], math.sqrt(zw[0] * zw[0] + zw[1] * zw[1]))

    unit_x = [1, 0, 0]
    Xc = np.array(unit_x).reshape(3, 1).astype(np.float64)
    #Xc(3, 1, CV_64FC1, unit_x)
    xw = np.dot(R_inv, Xc)
    #xw = (double)(Xw.data)
    xpan = [math.cos(pan), math.sin(pan), 0]
    print("[1,0,0] >> ", Xc, xw)

    roll = math.acos(xw[0] * xpan[0] + xw[1] * xpan[1] + xw[2] * xpan[2])
    if (xw[2] < 0): roll = -roll

    print("pan : ", pan)
    print("roll : ", roll)
    print("tilt : ", tilt)

    print(R)
    R_ = [[math.cos(pan), -math.sin(pan) * math.sin(tilt), -math.sin(pan) * math.cos(tilt)],
         [math.sin(pan), math.cos(pan) * math.sin(tilt), math.cos(pan) * math.cos(tilt)],
         [0, -math.cos(tilt), math.sin(tilt)]]

    print(R_)
extrinsic_dark(rvecs, tvecs)


''' ************************************* '''





################################################################
################################################################
################################################################

# matlab_toolbox를 이용해 k값에 들어갈 intrinsic parameters 구하기

# copy parameters to arrays
# A (Intrinsic Parameters) [fc, skew*fx, cx], [0, fy, cy], [0, 0, 1]

'''
cx = imageSizeX / 2                                                     ####### 수정해야함
cy = imageSizeY / 2

a = 1       # (auto_Cali 논문에선 NTSC(720*480)에선 0.91, PAL(720*576)에선 1.09)
fx = cx / tan(CameraFOV / 2)        # FOV in the horizontal direction
fy = a*fx                           # FOV in the vertical direction 넣고 fx랑 똑같이 구하던지
skew = 0

K = np.array([[fx,  skew,   cx  ],
              [0,   fy,     cy  ],
              [0,   0,      1   ]])

# Distortion Coefficients(kc) - 1st, 2nd
d = np.array([0.0110684, -0.0085019, 0, 0, 0]) # just use first two terms

image = cv2.imread('CALIB_C_8.jpg')
resize = cv2.resize(image, (640, 360))
images = glob.glob('CALIB_C_8.jpg')

for fname in images:

    # read image
    img = cv2.imread(fname)
    h, w = img.shape[:2]

    # undistort
    newcamera, roi = cv2.getOptimalNewCameraMatrix(K, d, (w,h), 0)
    newimg = cv2.undistort(img, K, d, None, newcamera)
    resize2 = cv2.resize(newimg, (640, 360))
    # save image
    newfname = fname+'.undis.tif'

    cv2.imwrite(newfname, newimg)

cv2.imwrite('Prior_img.jpg', image)
cv2.imshow("prior_img",resize)
cv2.imshow("new_img",resize2)
cv2.waitKey()
cv2.destroyAllWindows()

'''