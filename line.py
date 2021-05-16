import numpy as np
import math

class Point(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


class Line(object):  # 直线由两个点组成
    def __init__(self, p1=Point(0, 0), p2=Point(2, 2)):
        self.p1 = p1
        self.p2 = p2

    def distance_point_to_line(self, current_line, mainline):
        angle = self.get_cross_angle(current_line, mainline)
        sin_value = np.sin(angle * np.pi / 180)  # 其中current_line视为斜边
        long_edge = math.sqrt(  # 获取斜边长度
            math.pow(current_line.p2.x - current_line.p1.x, 2) + math.pow(current_line.p2.y - current_line.p1.y,
                                                                          2))  # 斜边长度
        distance = long_edge * sin_value
        return distance

    def get_cross_angle(self, l1, l2):
        arr_a = np.array([(l1.p2.x - l1.p1.x), (l1.p2.y - l1.p1.y)])  # 向量a
        arr_b = np.array([(l2.p2.x - l2.p1.x), (l2.p2.y - l2.p1.y)])  # 向量b
        cos_value = (float(arr_a.dot(arr_b)) / (np.sqrt(arr_a.dot(arr_a)) * np.sqrt(arr_b.dot(arr_b))))  # 注意转成浮点数运算
        return np.arccos(cos_value) * (180 / np.pi)  # 两个向量的夹角的角度， 余弦值：cos_value, np.cos(para), 其中para是弧度，不是角度

    def get_main_line(self, mask):
        # 获取最上面和最下面的contour的质心
        contour_list = get_contours(mask)
        c7_x, c7_y = get_centroid(contour_list[7])  # 最上面
        c0_x, c0_y = get_centroid(contour_list[0])  # 最下面

        # 获取串联两个质心，得到主线
        point1 = Point(c7_x, c7_y)
        point2 = Point(c0_x, c0_y)
        mainline = Line(point1, point2)
        return mainline

    #  求seg_img图中的直线与垂直方向的夹角
    def mainline_inclination_angle(self, seg_img):
        # 获取串联两个质心，得到主线
        mainline = self.get_main_line(seg_img)
        # 测试该函数，三角形边长：3,4,5
        mainline.p1.x = 0  # 列
        mainline.p1.y = 0
        mainline.p2.x = 3
        mainline.p2.y = 4
        # 获取参考线，这里用的是垂直方向的直线
        # base_line = Line.get_main_line(normal_mask)
        base_line = Line(Point(mainline.p1.x, mainline.p1.y),
                         Point(mainline.p1.x, mainline.p2.y))  # 同一列mainline.p1.x，行数随便
        # 获取两条线的夹角
        angle = mainline.get_cross_angle(mainline, base_line)
        return angle


def calculate_distance(cx, cy, base_line):
    '''
    点到直线的距离
    :param cx:
    :param cy:
    :param base_line: 这里的基准线是横着的方向, 由两个点组成
    :return:
    '''
    long_line = Line(Point(cx, cy), Point(base_line.p1.x, base_line.p1.y))  # 每个字的质心和基准线上的一个点组成的长边
    distance = long_line.distance_point_to_line(current_line=long_line, mainline=base_line)
    return distance