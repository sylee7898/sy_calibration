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
from __future__ import division, print_function, absolute_import

import argparse
import os
from typing import List, Any, Union

import cv2
import numpy as np
import matplotlib.pyplot as plt

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

import velocity as vel_py

# 근데 이거 이미 디텍션 된 정보 있는걸로 tracking함
# 영상내 det 파일(bbox만 있는)로 잘 돌아가나 보고 잘되면
# yolo로 bbox구해서 하기
'''
def detection_file_load(detection_file) :
    # the array is saved in the file geekfile.npy
    npy = np.load(detection_file)

    # the array is loaded into b
    print("npy is:")
    print(npy)
    print("row : ", len(npy))
    print("col[0] : ", len(npy[0]))         # col 다 138
'''

def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "img1")
    #detection_file = os.path.join(sequence_dir, "det", "det.txt")      # github에서 준 detection.npy 말고 영상내에 있는 bbox 정보로 되는지 확인
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])  # frameRate는 초당 몇frame인지 > 1장당 걸린시간은 1초/frameRate
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),    # video name
        "image_filenames": image_filenames,             # frames : frame 순서대로 한영상 다 들어가있음
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list



'''
run에서 detections
	[ bbox, confidence, feature ]
	confidence >= min_confidence 인것만 non_max_suppression한 i
	detection[i]만 detections로 
# 영상정보에 있는 det 에는 feature 정보 없음 
'''

def run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.
    IDnum   : int
        Tracking ID_num

    """


    # 프레임 사진 있는 위치, 영상번호 (seq_info["image_filenames"])
    # 프레임번호 (seq_info["image_filenames"][frame_idx])
    # 다 gather_sequence_info에 있음


    if 'y' == input("is there ID [y/n] : "):
        IDnum = int(input("ID_num for tracking : "))                 #########
        #range_down = input("ID tracking range_down : ")             #########
        #range_up = input("ID tracking range_up : ")                 #########
    else :
        IDnum = 0

    if 'y' == input("foot display [y/n] : "):       ############# y값 함수 구하고나면 발 위치보기
        foot_dis = True
    else:
        foot_dis = False



    seq_info = gather_sequence_info(sequence_dir, detection_file)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []
    target_h = []

    update_ms = seq_info["update_ms"]
    max_frame_idx = seq_info["max_frame_idx"]


    i = 0

    def frame_callback(vis, frame_idx):
        #print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        detections = create_detections(
            seq_info["detections"], frame_idx, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])  # (x, y, w, h)
        scores = np.array([d.confidence for d in detections])  # Detector confidence score
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        tracker.predict()
        tracker.update(detections)



        # Update visualization.
        if display or foot_dis:
            image = cv2.imread(
                seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            if IDnum == 0:
                vis.draw_detections(detections)
                vis.draw_trackers(tracker.tracks)
            else :      #찾는 ID 있을 때
                vis.draw_target_trackers(tracker.tracks, IDnum)
            if foot_dis:                            # Tracking 하는 ID만 보여주고 발 표시 !!!!!!!
                vis.draw_foot(tracker.tracks, IDnum)


        # h 저장
        h_file = os.path.dirname(output_file)  # result/text/
        with open(h_file + '/ID_h.txt', 'r') as f_hi:
            line_splits = [int(l.split(',')[1]) for l in f_hi.read().splitlines()[1:]]


        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            ############################################################### tracking 대신 bbox만 넣어주고 계산하기
            bbox = track.to_tlwh()

            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
            if int(track.track_id) == int(IDnum):
                #print("find ID-01")
                if bbox[1]+bbox[3] < seq_info["image_size"][0] :
                    target_h.append([track.track_id, bbox[1]+bbox[3], bbox[3]])  # id의 y값 별 h


                if(frame_idx >= 40) :   # start frame 이걸로 설정    ################################
                    # MOT16-02에서 할머니 멈추기 전까지가 260frame
                    endT = 117 * update_ms / 1000
                    # endT = 5
                    vel_py.foot_world(frame_idx, track.track_id, bbox, IDnum, update_ms, max_frame_idx, endT)
                    # velocity 추정 끝나면 count = 0으로 계산 그만시킴

                # foot 10 보기
                for i in range(10) :
                    if int(bbox[1]+bbox[3]) > line_splits[i]-1 and int(bbox[1]+bbox[3]) < line_splits[i]+1:
                        print(int(bbox[1] + bbox[3]))
                        vis.draw_foot(tracker.tracks, IDnum)




    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)

    ##### 영상 저장 ###########
    #video_output_dir = os.path.curdir + '/result/video'
    #video_filename = os.path.join(video_output_dir, "%s_all_tracking.avi" % (os.path.basename(sequence_dir)))       # video name은 seq_info["sequence_name"]
    #video_filename = os.path.join(video_output_dir, "%s_ID%s_tracking.avi" % (os.path.basename(sequence_dir), IDnum))
    #video_filename = os.path.join(video_output_dir, "%s_ID%s_foot 10.avi" % (os.path.basename(sequence_dir), IDnum))
    #if video_filename is not None:
    #    visualizer.viewer.enable_videowriter(video_filename)
    #########################
    visualizer.run(frame_callback)

    # Store results.
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]), file=f)
        # [frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]]
    f.close()



def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None,
        required=True)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)

    return parser.parse_args()


# 영상저장 껐

if __name__ == "__main__":
    args = parse_args()

    ###################

    run(
        args.sequence_dir, args.detection_file, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.display)

    #print(vel_py.foot_diff)
    print("velocity : ", vel_py.velocity)




#################################################################################
####################### 기존에 h 구하는 deep_sor_app 코드 ##########################
#################################################################################


#     ################################################ ID height ###############################################
#     # f_hi에 원하는 ID의 y, h 기
#
#     total_y = np.zeros(seq_info["image_size"][0])  # 이 ID가 지난 모든 y값 배열에 저장
#     linear_h_six: List[List[Union[float, Any]]] = []  # 6개의 y,h 저장
#     n = 0  # 6구간으로 나누고   10개평균내고 구간점프해서 중간중간 값들 구하기
#     h_mean = 0
#     y_mean = 0
#     sameID = len(target_h)  # ID 발견된 수
#
#     print(sameID)
#     six = int(sameID / 10)  # ID y를 8개로 나눔
#     print(six)
#     for hi in target_h:
#
#         y = int(hi[1])
#         total_y[y] += 1  # 해당 ID의 y값 분포 저장 >> y 히스토그램
#
#         n_pass = n % six
#         if n_pass < 10:  # 0-9는 평균구하고
#             h_mean += hi[2]
#             y_mean += hi[1]
#         elif n_pass == 10:  # 연속된 10개의 y값의 h평균을 구함
#             h_mean = h_mean / 10
#             y_mean = y_mean / 10
#         elif n_pass == (six - 1):
#             linear_h_six.append([h_mean, y_mean])  # x축이 height, y축이 y값
#             h_mean = 0
#             y_mean = 0
#         n += 1
#
#     # Store results.
#     h_file = os.path.dirname(output_file)  # result/text/
#     f_hi = open(h_file + '/ID_h.txt', 'w')
#     print('[h, y]', file=f_hi)
#     for row in linear_h_six:
#         print('%d, %d' % (row[0], row[1]), file=f_hi)
#     f_hi.close()
#     '''
#
#     ###################################################################
#     ########################################################################
#     ############################################################################
#
#     # 하우스홀더 변환 해보기
#     ##################################################
#
#
#
#     # linear_h_six
#     # ID03 - cam3
#     data = np.array([  # ID [h, y]
#         [151, 546],
#         [150, 524],
#         [145, 490],
#         [139, 446],
#         [136, 415],
#         [128, 360],
#         [126, 333],
#         [123, 292],
#         [115, 238],
#         [111, 203]])
#
#     # ID03 - cam1 : 좀 부정확
#     data2 = np.array([  # ID [h, y]
#         [146, 205],
#         [160, 309],
#         [164, 333],
#         [166, 372],
#         [172, 421],
#         [175, 458]])
#     '''
#     plt.scatter(data[:, 0], data[:, 1])        # 데이터 그래프에 점찍어보기
#
#     def qr_householder(A):
#         m, n = A.shape
#         Q = np.eye(m)  # Orthogonal transform so far
#         R = A.copy()  # Transformed matrix so far
#
#         for j in range(n):
#             # Find H = I - beta*u*u' to put zeros below R[j,j]
#             x = R[j:, j]
#             normx = np.linalg.norm(x)  # 벡터 길이 |x| 구하기
#             rho = -np.sign(x[0])  # sign 원소 부호판단 [양은1, 0, 음은-1]
#             u1 = x[0] - rho * normx  # x[0] + x[0]부호*x길이
#             u = x / u1
#             u[0] = 1
#             beta = -rho * u1 / normx
#
#             R[j:, :] = R[j:, :] - beta * np.outer(u, u).dot(R[j:, :])
#             Q[:, j:] = Q[:, j:] - beta * Q[:, j:].dot(np.outer(u, u))
#
#         return Q, R
#
#     #  [h,1] [a/b] = [y]
#     # > [a/b] = [h,1]T [y]
#     #         =   Q     b
#     m, n = data.shape
#     A = np.array([data[:, 0], np.ones(m)]).T  # [h, 1]T 얘를 Q,R로 분해
#     b = data[:, 1]  # [y]
#
#     # Q1A 로 [j:2]행렬 만들고
#     # Q2Q1A = Upper Trianglular Matrix가 됨
#     # Q=(Q2Q1)T
#     Q, R = qr_householder(A)  # A = Q R
#     b_hat = Q.T.dot(b)
#
#     R_upper = R[:n, :]
#     b_upper = b_hat[:n]
#
#     print(R_upper, b_upper)
#
#     x = np.linalg.solve(R_upper, b_upper)
#     slope, intercept = x
#     '''
#     #########################################################################
#     img = cv2.imread('/home/seungyeon/Desktop/git/neurvps/data/line/camera1_bg.jpg', cv2.IMREAD_COLOR)
#     # ******************** 그래프 *********************
#
#     # plot 다양하게 쓰기  https://howtothink.readthedocs.io/en/latest/PvL_H.html
#     '''
#     # 1번 y분포 히스토그램
#     plt.title('total_y')
#     x = seq_info["image_size"][0]  # 이미지 y값 : 1080
#     y = total_y
#     index = np.arange(x)
#
#     plt.bar(index, y)
#     plt.show()
#     '''
#     # 2번 y에 따른 h 값
#     plt.title('y-based height')
#
#     '''
#     # zip은 [(h,y), (h,y) ...] 을 [(h,h,h,h,..), (y,y,y,y,...)]으로 변환
#     #       *을 쓰면 이중리스트의 경우 리스트가 원소 차례대로 반환됨
#     # zip(list)나 list.T 같음 >> 바꿔서 연산량 측정해보기
#
#     trans = [x for x in zip(*linear_h_six)]
#
#     print(linear_h_six)
#     print("*************************************")
#     print(trans)
#     '''
#     #    plt.plot(trans[0], trans[1], 'b--')               # scatter은 점으로 표현, plot은 선으로 표현 / plot에서 'o'로 점표현가능
#     # 나중에 data는 점으로 그래프는 선으로 plt.plot(x_func=np.linspace(0, 10, 50), y_func=found_fit(x_func), label='$f(x) = 0.388 x^2$')
#     plt.xlabel('height')
#     plt.ylabel('image Y')
#     # plt.ylim(x/4, x*3/4)      #x는 히스토그램 구할때 쓴   seq_info["image_size"][0]로 이미지 높이 값
#
#     trans = [x for x in zip(*data)]
#     # plt.axis([0,10,0,20]) # X축 0-10, y축 0-20 으로 나눔
#     plt.plot(trans[0], trans[1], 'ro')
#
#
#     # 구간 10개로 나눴을 때 0,9 점 이어 일차방정식 만들기
#     ID_h0 = trans[0][2]
#     ID_y0 = trans[1][2]
#     ID_h9 = trans[0][9]
#     ID_y9 = trans[1][9]
#
#     # eq1 = a * ID_h0 + b - ID_y0
#     # eq2 = a * ID_h9 + b - ID_y9
#     a = (ID_y0 - ID_y9) / (ID_h0 - ID_h9)
#     b = ID_y0 - a * ID_h0
#
#     # solve((eq1, eq2), dict=True)            # 기울기 a와 절편 b를 구함
#     print("기울기 : %05f,  Y절편 : %05f" % (a, b))
#     # MOT16-01_ID01은 10구간일때 기울기 0.798, y절편 : 486
#
#     x = trans[0]
#     y = [a * v + b for v in x]
#     # index = np.arange(x)
#     plt.plot(x, y, 'b', label='y = a*h + b')
#
#     # 다른캠과 비교했을때 같은 y에 h값 몇인가
#     # cam3가 기울기 더 크니까 높아질수록 멀어짐
#     h = (img.shape[0] - b) / a
#     print("같은y에서 h = ", h)
#     plt.plot(h, img.shape[0], 'go')
#     ##########################################
#     # 다른캠 h 그래프 같이 그려넣기
#     '''
#     trans = [x for x in zip(*data2)]
#     # plt.axis([0,10,0,20]) # X축 0-10, y축 0-20 으로 나눔
#
#     plt.plot(trans[0], trans[1], 'bo')
#
#     # 구간 10개로 나눴을 때 0,9 점 이어 일차방정식 만들기
#     ID_h0 = trans[0][0]
#     ID_y0 = trans[1][0]
#     ID_h9 = trans[0][4]
#     ID_y9 = trans[1][4]
#
#     a = (ID_y0 - ID_y9) / (ID_h0 - ID_h9)
#     b = ID_y0 - a * ID_h0
#
#     print("기울기 : %05f,  Y절편 : %05f" % (a, b))
#
#     h = (img.shape[0] - b) / a
#     print("같은y에서 h = ", h)
#     plt.plot(h, img.shape[0], 'go')
#
#     x = trans[0]
#     x = [146, h]
#     y = [a * v + b for v in x]
#     # index = np.arange(x)
#     plt.plot(x, y, 'r', label='y = a*h + b')
#
#     '''
#     ###########################################
#
#
#
#     plt.show()
#     #cam3 제일 큰거에 맞출때 141/176 = 0.80
#
# # 플레이된 영상은 y 575까지 있음 (2배하면 1150)
# #######
#
#
# ##이제 키값 y별로 잘못된 값들 평균내서 적당한 키 방정식 구하기
# # ID같을때 target_h[0]이 같을때
#
#
# #    target_h[0][0]     #y별로 h어떻게 접근하지 2차배열인가 h=target_h[y].[2]
#
# # y값(hi[1])++ 해서 히스토그램만들기
# # y 10개 평균구하고 다음구간으로 넘기고 해서 h 10개 만들기
# # 일차방정식으로 만들어야하는데 우선은 점들만 찍어두기 > 점으로 표현하려면 여러 점 위치 더 파악
# # 값들가지고 일차방정식 최적화하기
#
# # 사진 위에 ID 발에 점찍기 bbox [[0]+[2]/2, [1]+[3]]
#     '''
#     plt.title('total_y')
#     x = len(target_h)       #이미지 h값
#     y = total_y
#     index = np.arange(x)
#
#     plt.bar(index, y)
#     plt.show()
#
# ############ 일차방정식 ##############
# import matplotlib.pylab as pl
# import numpy as np
# def draw_linear(m=1, x1=0, y1=0):
#     equ1 = a * target_h[3] + b -target_h[1]     # y절편 값을 구할 수 있을 때 변수에 저장
#     h_axis = np.arane(0, 300)                   #
#     y_axis = [(a*h_axis +equ1) for num in h_axis]   # y값을 그릴 그래프
#
#     pl.plot(h_axis, y_axis)
#     pl.grid()
#     pl.xlabel('height')
#     pl.ylabel('y')
#     pl.title('y-based height')
#     pl.savefig('y-based height.png')    # 이미지 파일 저장
#
#
#     equ1 = a * target_h[3] + b -target_h[1]
#     '''
