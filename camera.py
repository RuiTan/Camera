import time
import os
import cv2
import numpy as np
from image_registration import *
from edge_detection import *
from bounding_box import *
import queue

CSI_PORT = 'CSI'
USB_PORT = 'USB'
WIDTH = 1280
HEIGHT = 720
OUT_PATH = './capture/'
DIFF_PATH = './diff/'
BENCHMARK_PATH = './benchmark/'
TIME_FORMAT = '%Y-%m-%d-%H:%M:%S'
CELL_SIZE_THRESHOLD = 2
BENCHMARK = cv2.imread('./benchmark/benchmark.png')
BENCHMARK_CANNY = cv2.imread('./benchmark/benchmark_canny.png', cv2.CAP_OPENNI_GRAY_IMAGE)
BENCHMARK_GRAY = cv2.imread('./benchmark/benchmark_gray.png', cv2.CAP_OPENNI_GRAY_IMAGE)

if not os.path.exists(OUT_PATH): os.makedirs(OUT_PATH)
if not os.path.exists(DIFF_PATH): os.makedirs(DIFF_PATH)
if not os.path.exists(BENCHMARK_PATH): os.makedirs(BENCHMARK_PATH)


class Camera:
    # 初始化相机，针对CSI和USB两种接口有两种初始化方式
    def __init__(self, camera_type: str, camera_id: int, need_camera=True):
        super().__init__()
        if need_camera:
            if camera_type == CSI_PORT:
                self.camera = cv2.VideoCapture(self.gstreamer_pipeline(camera_id, flip_method=0), cv2.CAP_GSTREAMER)
            elif camera_type == USB_PORT:
                self.camera = cv2.VideoCapture(camera_id)
            if not self.camera.isOpened():
                print("Cannot open camera")
                exit()

    def diff(self, frame, benchmark, benchmark_canny, use_canny=True, use_gray=True):
        # sift特征匹配法与benchmark对齐
        # frame, _, _ = siftImageAlignment(frame, BENCHMARK)
        # 获取canny算子边缘检测结果
        frame_canny, _ = edge_detection(frame)
        # canny差异图
        canny_sub = filter(
            float_to_rgb(
                frame_canny.astype(float) - benchmark_canny.astype(float)
            ), threshold=THRESHOLD
        )
        # 灰度差异图
        gray_sub = filter(
            graySub(frame, benchmark), threshold=GRAY_SUB_THRESHOLD
        )
        # 融合canny和灰度差异
        sub = None
        if use_canny and use_gray:
            sub = canny_sub + gray_sub
            sub[sub == 255] = 0
            sub[sub == 510] = 255
        elif use_canny:
            sub = canny_sub
        elif use_gray:
            sub = gray_sub
        # 为变化簇标记颜色
        sub, count = self.color_sub(sub)
        if count > 0:
            current_time = self.timestamp()
            os.makedirs(DIFF_PATH + current_time)
            bb_features = bounding_box(sub, count)
            save_bb_feature(bb_features, DIFF_PATH + current_time + '/bb_features.json')
            bb_plt(frame, bb_features, DIFF_PATH + current_time + '/diff_boxes.png', 4.8, 6.4)
            # cv2.imwrite(frame_canny, DIFF_PATH, )
            print(current_time, '异物出现，数量为', count)
        else:
            print('当前没有异物')

    def capture(self):
        while True:
            # 获取当前帧
            ret, frame = self.camera.read()
            frame = cv2.flip(frame, 1)
            self.diff(frame, BENCHMARK, BENCHMARK_CANNY)

    # 首先去除像素点簇数量小于阈值的点，即噪声点；其次对于其余满足条件的变化簇进行颜色标记
    def color_sub(self, sub):
        q = queue.Queue()
        m, n = sub.shape[0], sub.shape[1]
        count = 0
        moveX = [0, 0, -1, 1]
        moveY = [-1, 1, 0, 0]
        for i in range(m):
            for j in range(n):
                if sub[i, j] == 255:
                    cell_size = 0
                    count += 1
                    sub[i, j] = count
                    q.put((i, j))
                    while not q.empty():
                        (x, y) = q.get()
                        for k in range(4):
                            xx = x + moveX[k]
                            yy = y + moveY[k]
                            if 0 <= xx < m and 0 <= yy < n and sub[xx, yy] == 254:
                                cell_size += 1
                                sub[xx, yy] = count
                                q.put((xx, yy))
                    if cell_size < CELL_SIZE_THRESHOLD:
                        sub[sub == count] = 0
                        count -= 1
        return sub, count

    def gstreamer_pipeline(
            self,
            camera_id=0,
            capture_width=WIDTH,
            capture_height=HEIGHT,
            display_width=WIDTH,
            display_height=HEIGHT,
            framerate=60,
            flip_method=0,
    ):
        return (
                "nvarguscamerasrc sensor-id=%d ! "
                "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, format=(string)NV12, framerate=(fraction)%d/1 ! "
                "nvvidconv flip-method=%d ! "
                "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
                "videoconvert ! "
                "video/x-raw, format=(string)BGR ! "
                "appsink"
                % (
                    camera_id,
                    capture_width,
                    capture_height,
                    framerate,
                    flip_method,
                    display_width,
                    display_height,
                )
        )

    def norm(self, img):
        max = np.max(img)
        min = np.min(img)
        img = (img - min) / (max - min)
        return img

    def timestamp(self, time_format=TIME_FORMAT):
        return time.strftime(time_format, time.localtime())

    def benchmark(self):
        ret, frame = self.camera.read()
        frame = cv2.flip(frame, 1)
        frame_canny, _ = edge_detection(frame)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        cv2.imwrite('./benchmark/benchmark2.png', frame)
        cv2.imwrite('./benchmark/benchmark_canny.png', frame_canny)
        cv2.imwrite('./benchmark/benchmark_gray.png', frame_gray)
