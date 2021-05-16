import cv2
from edge_detection import *
import os
from camera import *

def video_capture(video_full_path):
    cap = cv2.VideoCapture(video_full_path)
    print(cap.isOpened())
    frame_count = 1
    success = True
    while (success):
        success, frame = cap.read()
        print('Read a new frame: ', success)
        params = []
        # params.append(cv.CV_IMWRITE_PXM_BINARY)
        params.append(1)
        cv2.imwrite("video_capture/video" + "_%d.jpg" % frame_count, frame, params)

        frame_count = frame_count + 1

    cap.release()


def compare_video_capture():
    data_path = 'video_capture/'
    benchmark = cv2.imread(data_path + 'video_2.jpg')
    benchmark_canny, _ = edge_detection(benchmark)
    imgs = os.listdir(data_path)
    for i in imgs:
        img = cv2.imread(data_path + i)
        camera = Camera('USB', 0, False)
        camera.diff(img, benchmark, benchmark_canny, False, True)

compare_video_capture()