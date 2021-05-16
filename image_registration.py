import numpy as np
import cv2
from edge_detection import edge_detection

DATA_PATH = './capture/'

def sift_kp(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)
    kp_image = cv2.drawKeypoints(gray_image, kp, None)
    return kp_image, kp, des


def get_good_match(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good


def siftImageAlignment(img1, img2):
    _, kp1, des1 = sift_kp(img1)
    _, kp2, des2 = sift_kp(img2)
    goodMatch = get_good_match(des1, des2)
    imgOut = img1
    H = None
    status = None
    if len(goodMatch) > 4:
        ptsA = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ransacReprojThreshold = 4
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold);
        # 其中H为求得的单应性矩阵矩阵
        # status则返回一个列表来表征匹配成功的特征点。
        # ptsA,ptsB为关键点
        # cv2.RANSAC, ransacReprojThreshold这两个参数与RANSAC有关
        imgOut = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]),
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return imgOut, H, status


# if __name__ == '__main__':
#     img1 = cv2.imread(DATA_PATH + 'sift1.jpeg')
#     img2 = cv2.imread(DATA_PATH + 'sift2.jpeg')
#
#     result, _, _ = siftImageAlignment(img1, img2)
#     cv2.imwrite(DATA_PATH + 'sift_result.jpeg', result)
#     # allImg = np.concatenate((img1, img2, result), axis=1)
#     # cv2.imwrite(DATA_PATH + 'registration.jpeg', allImg)
#     edge_detection('sift1.jpeg')
#     edge_detection('sift_result.jpeg')
#     old = cv2.imread(DATA_PATH + 'Canny-sift1.jpeg')
#     new = cv2.imread(DATA_PATH + 'Canny-sift_result.jpeg')
#     sub = (old.astype(float)-new.astype(float))
#     cv2.imshow('old', old)
#     cv2.imshow('new', new)
#     cv2.imshow('sub', sub)
#     cv2.waitKey(0)
