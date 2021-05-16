import cv2
from matplotlib import pyplot as plt
import numpy as np
import json


def bounding_box(img, count):
    bb_features = []
    for i in range(1, count+1):
        xs, ys = np.where(img == i)
        begin_x = np.min(xs)
        begin_y = np.min(ys)
        len_x = np.max(xs) - np.min(xs)
        len_y = np.max(ys) - np.min(ys)
        bb_feature = {
            'x': int(begin_y),
            'y': int(begin_x),
            'len_x': int(len_y),
            'len_y': int(len_x)
        }
        bb_features.append(bb_feature)
    return bb_features


def save_bb_feature(bb_features, feature_path):
    feature_file = open(feature_path, 'w')
    json.dump(bb_features, feature_file)
    feature_file.close()


def bb_plt(im, bb_features, save_path, height=14.4, width=10.8):
    # im = im[:, :, (2, 1, 0)]
    # fig = plt.figure(figsize=(height, width))
    # ax = fig.add_subplot()
    # ax.imshow(im, aspect='equal')
    for bbox in bb_features:
        x_min = bbox['x']
        y_min = bbox['y']
        x_max = x_min + bbox['len_x']
        y_max = y_min + bbox['len_y']
        cv2.rectangle(im, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv2.imwrite(save_path, im)
    #     ax.add_patch(
    #         plt.Rectangle((bbox['x'], bbox['y']),
    #                       bbox['len_x'],
    #                       bbox['len_y'], fill=False,
    #                       edgecolor='red', linewidth=3.5)
    #     )
    # plt.axis('off')
    # plt.tight_layout()
    # plt.savefig(save_path)  # path为你将保存图片的路径
    # plt.draw()
    # plt.show()
