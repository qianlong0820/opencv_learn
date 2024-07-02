import cv2 as cv
import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt
import uuid
import json
import logging
import math
import os
import pickle
import re
import shutil
import uuid
from pathlib import Path
import cv2
import numpy as np
import imutils
import easyocr
from PIL import Image
from difflib import SequenceMatcher
from paddleocr import PaddleOCR
import difflib
OCR = PaddleOCR(lang="ch", enable_mkldnn=False)


# 读取图像
def get_perspective_transform(image, target_contour):
    # 提取目标区域的四个角点
    points = cv2.approxPolyDP(target_contour, 0.02 * cv2.arcLength(target_contour, True), True)

    # 找到与原图左上角最近的点的坐标
    nearest_to_origin = min(points, key=lambda point: np.linalg.norm(point[0]))

    # 找到与原图右上角最近的点的坐标
    nearest_to_top_right = min(points, key=lambda point: np.linalg.norm(point[0] - np.array([image.shape[1], 0])))

    nearest_to_bottom_right = min(points, key=lambda point: np.linalg.norm(
        point[0] - np.array([image.shape[1], image.shape[0]])))

    # 找到与原图左下角最近的点的坐标
    nearest_to_bottom_left = min(points, key=lambda point: np.linalg.norm(point[0] - np.array([0, image.shape[0]])))

    # 按照你的逻辑提取排序后的四个角点
    top_left = nearest_to_origin[0]
    top_right = nearest_to_top_right[0]
    bottom_right = nearest_to_bottom_right[0]
    bottom_left = nearest_to_bottom_left[0]

    # 定义目标区域矫正后的四个角点
    src_points = np.float32([top_left, top_right, bottom_right, bottom_left])
    print(src_points)
    # 定义目标区域矫正后的四个角点
    # dst_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    width = top_right[0] - top_left[0]
    height = bottom_right[1] - top_right[1]
    dst_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    print(dst_points)
    # 计算透视变换矩阵
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    return perspective_matrix, width, height, top_left


def correction(image_path):
    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 设置黑色阈值来提取黑色区域
    black_thresh = cv2.adaptiveThreshold(blurred, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)

    # 查找轮廓
    contours, _ = cv2.findContours(black_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imwrite("/home/ya/mapdata/test/2.jpg", black_thresh)
    # areas = [cv2.contourArea(contour) for contour in contours]
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
    # print(f"轮廓数量{len(sorted_contours)}")

    one = cv2.contourArea(sorted_contours[0])
    three = cv2.contourArea(sorted_contours[2])
    # print(cv2.contourArea(sorted_contours[0]))
    # print(cv2.contourArea(sorted_contours[1]))
    # print(cv2.contourArea(sorted_contours[2]))

    # 创建一个全白的空白图像
    contour_image = np.ones_like(image) * 255

    # 在空白图像上绘制第一大和第二大面积的边框
    cv2.drawContours(contour_image, sorted_contours, -1, (0, 0, 255), 2)

    # 保存绘制了边框的图像
    cv2.imwrite("2.jpg", contour_image)

    # 检查第一大和第三大轮廓的面积差是否超过百分之50
    if len(sorted_contours) >= 3 and (one - three) / one > 0.5:
        max_contour = sorted_contours[0]
    else:
        max_contour = sorted_contours[2]

    total_area = image.shape[0] * image.shape[1]
    max_contour_area = cv2.contourArea(max_contour)
    # 计算 max_contour 占原始图像面积的比例
    ratio = max_contour_area / total_area
    # import pdb;pdb.set_trace()
    # # 绘制最大轮廓
    # cv2.drawContours(image, [max_contour], -1, (0, 0, 255), 2)

    # 计算透视变换矩阵
    perspective_matrix, width, height, top_left = get_perspective_transform(image, max_contour)
    # 应用透视变换
    warped_image = cv2.warpPerspective(image, perspective_matrix, (width, height))
    print(width,height)

    # cv2.imshow('Warped Image', warped_image)
    # cv2.waitKey(0)

    image_name = f"{uuid.uuid4()}.jpg"

    new_image_path = os.path.join("./resource", image_name)

    cv2.imwrite(new_image_path, warped_image)

    return new_image_path, ratio, width, height, [top_left[0],top_left[1]+image.shape[0]]


def ocr(new_image_path):
    OCR = PaddleOCR(lang="ch", enable_mkldnn=False)
    result = OCR.ocr(new_image_path, cls=True)
    rects = []
    texts = []
    # 遍历OCR识别结果和存储文本进行对比
    for lines in result:
        for line in lines:
            # ocr_text = ''.join(c for c in line[1][0] if c.isalnum())
            rects.append([int(float(line[0][0][0])), int(float(line[0][0][1])), int(float(line[0][2][0])),
                          int(float(line[0][2][1]))])
            text = ''.join(c for c in line[1][0] if c.isalnum())
            texts.append(text)  # 提取文本信息
    # correction("","","")

    print(texts)


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def ocrsample(img_path):
    folder_path = "/Users/du/work/py/pythonProject/opencv_learn/ocrsample/"
    if os.path.exists(folder_path):
        # 清空文件夹
        shutil.rmtree(folder_path)
        # 重新创建空文件夹
        os.makedirs(folder_path)
    image = cv2.imread(img_path)
    # 底部ocr保存
    # 获取图像的高度和宽度
    height, width = image.shape[:2]

    # 计算下50%部分的起始行
    start_row = int(height * 0.5)

    # 提取下40%部分的图像
    bottom_50_percent_image = image[start_row:height, :]

    # 显示下50%部分的图像
    cv2.imwrite("ocrsample/bottom_50.jpg", bottom_50_percent_image)
    # 基础高度
    base_p_h = image.shape[0]

    new_image_path, _, width, height, top_left = correction("ocrsample/bottom_50.jpg")
    shutil.move(new_image_path, "ocrsample/bottom.jpg")
    bottomImageResult = OCR.ocr("ocrsample/bottom.jpg", cls=True)
    bottomImageResult = bottomImageResult[0]
    with open('ocrsample/ocr_bottom_result.txt', 'w', encoding='utf-8') as f:
        for line in bottomImageResult:
            text = ''.join(c for c in line[1][0] if c.isalnum())
            box_str = ' '.join(str(coord) for point in line[0] for coord in point)
            # 将结果写入到txt文件中
            f.write(f"{box_str} {text}\n")

    with open('ocrsample/ocr_bottom_result.txt', 'r') as file:
        lines = file.readlines()
        rects = []
        texts = []
        for line in lines:
            # print(line)
            data = line.strip().split(' ')
            # print(data)
            rect = [int(float(data[0])), int(float(data[1])), int(float(data[4])), int(float(data[5]))]
            rects.append(rect)
            texts.append(data[-1])

        connector = BoxesConnector(rects, texts, width, max_dist=15, overlap_threshold=0.4)
        new_rects, new_texts = connector.connect_boxes()
    with open('ocrsample/ocr_bottom_result.txt', 'w') as file:
        for rect, text in new_texts.items():
            # 将矩形框和文本以指定格式写入文件
            file.write(
                f"{rect[0]} {rect[1]} {rect[2]} {rect[3]} {text} {rect[0] + top_left[0]} {rect[1] + top_left[1]} {rect[2] + top_left[0]} {rect[3] + top_left[1]}\n")

    result = OCR.ocr(img_path, cls=True)
    result = result[0]
    width = image.shape[1]
    # print(result)

    # 读取另外一个文本文件的前两行
    with open('industrialLabels/sample/sampleTxt.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if len(lines) > 2:
            lines = [''.join(c for c in line if c.isalnum()) for line in lines[:2]]
            last_line = ''.join(c for c in lines[-1] if c.isalnum())
            lines.append(last_line)
        lines = [''.join(c for c in line if c.isalnum()) for line in lines[:2]]
        # print(lines)

    with open('ocrsample/ocr_result.txt', 'w', encoding='utf-8') as f:
        for line in result:
            text = ''.join(c for c in line[1][0] if c.isalnum())
            for other_line in lines:
                similarity = similar(text, other_line)
                if similarity >= 0.7:
                    # 替换掉OCR读出的内容
                    text = other_line
                    # print(text)
            # 获取文本内容并去除所有符号（包括标点符号和空格）
            if "Water" in text or "NSF" in text or "An" in text:
                print(text)
                print("跳过")
                continue
            # 获取文本框坐标并转换为字符串
            box_str = ' '.join(str(coord) for point in line[0] for coord in point)
            # 将结果写入到txt文件中
            f.write(f"{box_str} {text}\n")

    with open('ocrsample/ocr_result.txt', 'r') as file:
        lines = file.readlines()

        rects = []
        texts = []

        for line in lines:
            print(line)
            data = line.strip().split(' ')
            print(data)
            rect = [int(float(data[0])), int(float(data[1])), int(float(data[4])), int(float(data[5]))]
            rects.append(rect)
            texts.append(data[-1])

        connector = BoxesConnector(rects, texts, width, max_dist=15, overlap_threshold=0.4)
        new_rects, new_texts = connector.connect_boxes()
    with open('ocrsample/ocr_result.txt', 'w') as file:
        for rect, text in new_texts.items():
            # 将矩形框和文本以指定格式写入文件
            file.write(f"{rect[0]} {rect[1]} {rect[2]} {rect[3]} {text}\n")
    # # 排除bottom的内容
    #
    # # 读取ocr_result.txt内容
    # with open('ocrsample/ocr_result.txt', 'r', encoding='utf-8') as f:
    #     ocr_result_lines = f.readlines()
    #
    # # 读取ocr_bottom_result.txt内容
    # with open('ocrsample/ocr_bottom_result.txt', 'r', encoding='utf-8') as f:
    #     ocr_bottom_result_lines = f.readlines()
    #
    # # 提取ocr_bottom_result.txt每一行的前五个字段作为关键信息
    # filtered_lines = []
    # for lineb in ocr_bottom_result_lines:
    #     bottomLine = lineb.split(' ')
    #     key = bottomLine[4].strip()
    #     # 过滤掉ocr_result.txt中包含ocr_bottom_result.txt中关键信息的行
    #     for line in ocr_result_lines:
    #         should_exclude = False
    #         ol = line.split(' ')[4].strip()
    #         if levenshtein_distance(ol, key) > 0.8:
    #             filtered_lines.append(line)
    #
    # filtered_ocr_result_lines = [line for line in ocr_result_lines if line not in filtered_lines]
    # # 将筛选后的结果写回ocr_result.txt
    # with open('ocrsample/ocr_result.txt', 'w', encoding='utf-8') as f:
    #     f.writelines(filtered_ocr_result_lines)



def get_rect_points(text_boxes):
    x1 = np.min(text_boxes[:, 0])
    y1 = np.min(text_boxes[:, 1])
    x2 = np.max(text_boxes[:, 2])
    y2 = np.max(text_boxes[:, 3])
    return [x1, y1, x2, y2]
# 定义一个函数来计算Levenshtein距离
def levenshtein_distance(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).ratio()

class BoxesConnector(object):
    def __init__(self, rects, texts, imageW, max_dist=None, overlap_threshold=None):
        self.rects = np.array(rects)
        self.texts = np.array(texts)  # 添加文本信息
        self.imageW = imageW
        self.max_dist = max_dist
        self.overlap_threshold = overlap_threshold
        self.graph = np.zeros((self.rects.shape[0], self.rects.shape[0]))

        self.r_index = [[] for _ in range(imageW)]
        for index, rect in enumerate(rects):
            if int(rect[0]) < imageW:
                self.r_index[int(rect[0])].append(index)
            else:
                self.r_index[imageW - 1].append(index)

    def calc_overlap_for_Yaxis(self, index1, index2):
        # 计算两个框在Y轴方向的重合度(Y轴错位程度)
        height1 = self.rects[index1][3] - self.rects[index1][1]
        height2 = self.rects[index2][3] - self.rects[index2][1]
        y0 = max(self.rects[index1][1], self.rects[index2][1])
        y1 = min(self.rects[index1][3], self.rects[index2][3])
        # print('y1', y1)
        Yaxis_overlap = max(0, y1 - y0) / max(height1, height2)

        # print('Yaxis_overlap', Yaxis_overlap)
        return Yaxis_overlap

    def get_proposal(self, index):
        rect = self.rects[index]
        # print('rect',rect)

        for left in range(rect[0] + 1, min(self.imageW - 1, rect[2] + self.max_dist)):
            # print('left',left)
            for idx in self.r_index[left]:
                # print('58796402',idx)
                # index: 第index个rect(被比较rect)
                # idx: 第idx个rect的x轴起始坐标大于被比较rect的x轴起始坐标(+max_dist)且小于被比较rect的x轴终点坐标(+max_dist)
                if self.calc_overlap_for_Yaxis(index, idx) > self.overlap_threshold:
                    return idx

        return -1

    def sub_graphs_connected(self):
        sub_graphs = []  # 相当于一个堆栈
        for index in range(self.graph.shape[0]):
            # 第index列全为0且第index行存在非0
            if not self.graph[:, index].any() and self.graph[index, :].any():  # 优先级是not > and > or
                v = index
                # print('v',v)
                sub_graphs.append([v])
                # print('sub_graphs', sub_graphs)
                # 级联多个框(大于等于2个)
                # print('self.graph[v, :]', self.graph[v, :])
                while self.graph[v, :].any():
                    v = np.where(self.graph[v, :])[0][
                        0]  # np.where(self.graph[v, :])：(array([5], dtype=int64),)  np.where(self.graph[v, :])[0]：[5]
                    # print('v11',v)
                    sub_graphs[-1].append(v)
                    # print('sub_graphs11', sub_graphs)
        return sub_graphs

    def connect_boxes(self):
        for idx, _ in enumerate(self.rects):
            proposal = self.get_proposal(idx)
            if proposal >= 0:
                self.graph[idx][proposal] = 1

        sub_graphs = self.sub_graphs_connected()

        set_element = set([y for x in sub_graphs for y in x])
        for idx, _ in enumerate(self.rects):
            if idx not in set_element:
                sub_graphs.append([idx])

        result_rects = []
        result_texts = {}  # 使用字典来存储文本信息，以便与矩形框对应
        for sub_graph in sub_graphs:
            rect_set = self.rects[list(sub_graph)]
            rect_set = get_rect_points(rect_set)
            result_rects.append(rect_set)

            # 合并文本信息
            text_set = self.texts[list(sub_graph)]
            text = ''.join(str(text) for text in text_set)
            result_texts[tuple(rect_set)] = text  # 使用矩形框的元组作为键，文本作为值

        return np.array(result_rects), result_texts


ocrsample('./industrialLabels/sample/sampleSplit.jpg')

# new_image_path, ratio = correction("/Users/du/Desktop/industrialLabels/2024-06-21_15-24-21.jpg", "", "")

# new_image_path, ratio = correction("/Users/du/Desktop/industrialLabels/2024-06-21_15-15-10.jpg", "", "")
# ocr(new_image_path)
#
# new_image_path, ratio = correction("/Users/du/Desktop/industrialLabels/2024-06-21_15-14-44.jpg", "", "")
# ocr(new_image_path)
#
# print('/Users/du/Desktop/industrialLabels/2024-06-21_15-14-25.jpg')
# new_image_path, ratio = correction("/Users/du/Desktop/industrialLabels/2024-06-21_15-14-25.jpg", "", "")
# ocr(new_image_path)
# 读取图像
# img_path = './222.jpg'
# # # 显示下40%部分的图像
# print(img_path)
# correction(img_path, "", "")

# image = cv2.imread('/Users/du/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/acb79dac98c4ef1b789ef27d4c1df0a7/Message/MessageTemp/e458c37f29f662d4f41a9c6bc3ed8997/File/industrialLabels/sample/sampleSplit.jpg')
#
# 获取图像的高度和宽度
# height, width = image.shape[:2]
#
# # 计算下40%部分的起始行
# start_row = int(height * 0.6)
#
# # 提取下40%部分的图像
# bottom_40_percent_image = image[start_row:height, :]
#
# # 显示下40%部分的图像
# cv2.imwrite("222.jpg",bottom_40_percent_image)
