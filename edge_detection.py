import cv2
from tqdm import tqdm
from line import *
import time

DATA_PATH = './capture/'
THRESHOLD = 0.1
GRAY_SUB_THRESHOLD = 0.5


def edge_detection(img):
    gaussian_blur = cv2.GaussianBlur(img, (5, 5), 0)
    canny = cv2.Canny(gaussian_blur, 35, 125)

    _, threshold = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    gradient = cv2.morphologyEx(threshold, cv2.MORPH_GRADIENT, kernel)
    return canny, gradient


def graySub(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(float)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(float)
    sub = float_to_rgb(img1_gray - img2_gray)
    return sub


def float_to_rgb(img):
    img = np.abs(img)
    max = np.max(img)
    min = np.min(img)
    img = (img - min) / (max - min)
    img = (img * 255).astype(np.uint8)
    return img


def filter(img, threshold):
    img[img >= (255 * threshold)] = 255
    img[img < (255 * threshold)] = 0
    return img.astype(np.uint8)


def norm(img):
    max = np.max(img)
    min = np.min(img)
    img = (img - min) / (max - min)
    return img


def distance_to_camera(knownWidth, focalLength, perWidth):
    return (knownWidth * focalLength) / perWidth


def find_marker(image):
    canny, _ = edge_detection(image)
    _, cnts, _ = cv2.findContours(canny.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea)
    cs = []
    if len(cnts) <= 2:
        cs.append(cv2.minAreaRect(cnts[-1]))
        cs.append(cv2.minAreaRect(cnts[-1]))
        return cs, False
    else:
        cs.append(cv2.minAreaRect(cnts[-1]))
        cs.append(cv2.minAreaRect(cnts[-3]))
        return cs, True


def calc_pixel_length(length, pixels):
    return length / pixels


def mark_video_gap(gap_save_path, video_out_path, video_type=0, video_path='', read_fps=0, camera_id=0, time_interval=1):
    '''
    将视频流读取至程序中
    :param gap_save_path: 表示缝隙宽度输出的文件
    :param video_out_path: 表示视频文件的输出位置
    :param video_type: 0表示读取视频，1表示实时读取摄像头
    :param video_path: 在video_type=0时有效，指视频路径
    :param read_fps: 在video_type=1时有效，指读取的帧数
    :param camera_id: 在video_type=1时有效，指摄像头的id
    :param time_interval: 指每隔几帧保留一次有效帧
    :return:
    '''
    # frames = []
    gap_file = open(gap_save_path, 'w')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    count = 0
    start_time = time.time()
    if video_type == 0:
        print('=====================================================')
        print('开始拆分视频...')
        vid_cap = cv2.VideoCapture(video_path)
        video_writer = cv2.VideoWriter(video_out_path, fourcc, vid_cap.get(5), (int(vid_cap.get(3)), int(vid_cap.get(4))), True)
        success, image = vid_cap.read()
        while success:
            success, image = vid_cap.read()
            count += 1
            # if count % time_interval == 0:
            #     frames.append(image)
            if image is not None:
                image, min1, min2 = process_image(image)
                gap_file.write(str(format(min1, '.1f')) + ' ' + str(format(min2, '.1f')) + '\n')
                video_writer.write(image)
        print('视频拆分完成...')
        print('=====================================================')
    else:
        print('=====================================================')
        print('开始读取摄像头，按\'Q\'键停止读取...')
        vid_cap = cv2.VideoCapture(camera_id)
        success, image = vid_cap.read()
        video_writer = cv2.VideoWriter(video_path, fourcc, vid_cap.get(5), (int(vid_cap.get(3)), int(vid_cap.get(4))),
                                      True)
        print(time.time())
        while success:
            success, image = vid_cap.read()
            count += 1
            # if count % time_interval == 0:
            #     frames.append(image)
            if image is not None:
                image, min1, min2 = process_image(image)
                gap_file.write(str(format(min1, '.1f')) + ' ' + str(format(min2, '.1f')) + '\n')
                video_writer.write(image)
            if read_fps != 0:
                time.sleep(1.0 / float(read_fps))
            if cv2.waitKey(1) == ord('q'):
                break
        print('影像读取完成...')
        print('=====================================================')
    video_writer.release()
    end_time = time.time()
    print('共计花费：', end_time-start_time, 's')
    print('共计处理：', count, '帧')
    print('程序处理速率大约为：', int(count/(end_time-start_time)), 'fps')


# def frame2video(images, video_path, frame_size, fps=20):
#     print('=====================================================')
#     print('开始导出视频...')
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     videoWriter = cv2.VideoWriter(video_path, fourcc, fps, frame_size, True)
#     for i in images:
#         if i is not None:
#             videoWriter.write(i)
#     videoWriter.release()
#     print('视频导出成功...')
#     print('=====================================================')


def process_image(img, pixel_length=0.3, show=False, direction=0):
    markers, flag = find_marker(img)
    boxes = []
    for marker in markers:
        box = np.int0(cv2.boxPoints(marker))
        boxes.append(box)
        cv2.drawContours(img, [box], -1, (0, 255, 0), 2)

    marker0 = boxes[0]
    marker1 = boxes[1]

    if direction == 0:
        start = 0
    else:
        start = 1
    len_ = None
    point_ = None
    for i in [start, start + 2]:
        marker0_0 = Point(marker0[i][0], marker0[i][1])
        marker0_1 = Point(marker0[(i + 1) % 4][0], marker0[(i + 1) % 4][1])
        line = Line(marker0_0, marker0_1)
        marker1_0 = Point(marker1[i % 4][0], marker1[(i) % 4][1])
        marker1_1 = Point(marker1[(i + 1) % 4][0], marker1[(i + 1) % 4][1])
        marker1_2 = Point(marker1[(i + 2) % 4][0], marker1[(i + 2) % 4][1])
        marker1_3 = Point(marker1[(i + 3) % 4][0], marker1[(i + 3) % 4][1])
        len1 = calculate_distance(marker1_0.x, marker1_0.y, line)
        len2 = calculate_distance(marker1_1.x, marker1_1.y, line)
        len3 = calculate_distance(marker1_2.x, marker1_2.y, line)
        len4 = calculate_distance(marker1_3.x, marker1_3.y, line)
        if len_ is None or min(len_) > min([len1, len2, len3, len4]):
            len_ = [len1, len2, len3, len4]
            point_ = [marker1_0, marker1_1, marker1_2, marker1_3]

    indices = np.argsort(len_)
    len_ = [len_[indices[0]]*pixel_length, len_[indices[1]]*pixel_length]
    point_ = [point_[indices[0]], point_[indices[1]]]

    cv2.circle(img, (point_[0].x, point_[0].y), 10, (0, 255, 0), 5)
    cv2.circle(img, (point_[1].x, point_[1].y), 10, (0, 255, 0), 5)
    cv2.putText(img, 'Min gap size: ' + str(format(len_[0], '.1f')) + 'mm', (point_[0].x+50, point_[0].y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 0), 3)
    cv2.putText(img, 'Max gap size: ' + str(format(len_[1], '.1f')) + 'mm', (point_[1].x+50, point_[1].y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 0), 3)
    if show:
        cv2.imshow("img", img)
        cv2.waitKey(0)
    return img, len1, len2

if __name__ == '__main__':
    target = 'rotate_boxes'
    mark_video_gap('benchmark/' + target + '.txt', video_out_path='benchmark/' + target + '.avi', video_path='benchmark/' + target + '.mp4')
