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
import time

from PIL import Image
from difflib import SequenceMatcher
import os
import numpy as np
from PIL import Image
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications import ResNet152V2
# from tensorflow.keras.applications.resnet_v2 import preprocess_input

from sklearn.metrics.pairwise import cosine_similarity

# resnet152v2 = ResNet152V2(weights='imagenet', include_top=False,
#                           pooling='max', input_shape=(640, 640, 3))

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# from mobile_sam import sam_model_registry, SamPredictor
import torch
from paddleocr import PaddleOCR
import threading

OCR = PaddleOCR(lang="ch", enable_mkldnn=False)

# from pymodbus.client.sync import ModbusTcpClient
from skimage.metrics import structural_similarity as ssim


#
#
# # Modbus TCP服务器的IP地址和端口号
# server_ip = '192.168.0.10'
# server_port = 502
#
# # 创建一个Modbus TCP客户端
# client = ModbusTcpClient(server_ip, port=server_port)

# 连接到Modbus设备
def detect_indicia(org_img: str, org_path: str):
    modelpath = Path(org_path).joinpath("simple_ocr.pickle")
    jsonpath = Path(org_path).joinpath("simple_index2label.json")
    with open(modelpath, "rb") as f:
        model = pickle.load(f)
    with open(jsonpath, "r") as f:
        index2label = json.load(f)

    max_height = 30
    max_width = 30

    def make_im_template(im):
        template = np.zeros((max_height, max_width))
        offset_height = int((max_height - im.shape[0]) / 2)
        offset_width = int((max_width - im.shape[1]) / 2)
        template[offset_height:offset_height + im.shape[0], offset_width:offset_width + im.shape[1]] = im
        return template.reshape(max_height * max_width)

    def split_letters(im):
        # 2值化
        img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # 自动选择分割灰度等级的阈值
        # img_temp = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # 根据灰度分析给定阈值
        img_temp = cv2.threshold(img_gray, 110, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # # 纵向膨胀
        kernel = np.ones((3, 1), np.uint8)
        # # 白色是1，黑色是0
        dilation = cv2.dilate(img_temp, kernel, iterations=1)
        # plt.imshow(dilation, cmap="gray", norm=NoNorm())
        # plt.show()

        cnts = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # 过滤太小的
        # for i, c in enumerate(cnts):
        #     x, y, w, h = cv2.boundingRect(c)
        #     if h < 15:
        #         cv2.fillPoly(thresh, pts=[c], color=(0))
        # 画轮廓框
        # cv2.drawContours(im, cnts, -1, (255, 0, 0), 1)
        # plt.imshow(im)
        # plt.show()

        # 分割
        char_list = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if 3 < w < 10 and h < 10:
                continue
            if 10 < w and h < 15:
                continue
            if w > 25:
                continue
            if h > 25:
                continue
            cropImg = img_temp[y:y + h, x:x + w]
            char_list.append((x, cropImg))
            # print(char_list)
        return char_list

    def ocr_recognize(fname):
        im = cv2.imread(fname)
        char_list = split_letters(im)

        result = []
        for ch in char_list:
            res = model.predict([make_im_template(ch[1])])[0]  # 识别单个结果
            result.append({
                "x": ch[0],
                "label": index2label[str(res)]
            })
        result.sort(key=lambda k: (k.get('x', 0)), reverse=False)  # 因为是单行的，所以只需要通过x坐标进行排序。

        return "".join([it["label"] for it in result])

    string = ocr_recognize(org_img)
    print(string)
    if '\'\'' not in string:
        str2 = []
        count = []
        for i in string:
            str2.append(i)
            if i == '\'':
                count.append('\'')
                if len(count) % 2 == 0:
                    str2.append('\'')
        string = ''.join(str2)
    # a = len(string) / 3
    # separated = "      ".join(re.findall('.{%d}' % a, string))
    # print(separated)
    aa = string.split("\'\'")
    del aa[-1]
    logger.debug(f'数组长度{len(aa)}')
    degree = re.findall('(\d+)°', f"{aa}")
    minute = re.findall('°(\d+)\'', f"{aa}")
    second = re.findall('\'(\d+)', f"{aa}")
    logger.debug(f'度{degree},分{minute},秒{second}')
    for index, i in enumerate(second):
        if int(i) > 60:
            print(i[0:2])
            second[index] = i[0:2]
            print(f'第一次处理后的{second}')
            if int(second[index]) > 60:
                second[index] = i[0:1]
    bb = []
    for i in range(len(aa)):
        result = float(degree[i]) + float(minute[i]) / 60. + float(second[i]) / 3600.
        bb.append(result)
    return bb


def cutimage(org_img: str, out_path: str):
    # 打开一张图
    img = Image.open(org_img)
    # 图片尺寸
    img_size = img.size
    h = img_size[1]  # 图片高度
    print(h)
    w = img_size[0]  # 图片宽度
    print(w)
    # 开始截取
    region = img.crop((203, 304, 255, 2010))
    region1 = img.crop((253, 250, 3108, 305))
    regionall = img.crop((255, 306, 3108, 2010))
    regionup = img.crop((253, 297, 3108, 305))
    regionleft = img.crop((242, 304, 255, 2010))
    rotate = np.rot90(region, -1)
    rotatelabel = np.rot90(regionleft, -1)
    left = out_path.joinpath('left.png')
    up = out_path.joinpath('up.png')
    all = out_path.joinpath('all.png')
    leftlabel = out_path.joinpath('leftlabel.png')
    uplabel = out_path.joinpath('uplabel.png')
    cv2.imwrite(f'{left}', rotate)
    region1.save(f'{up}')
    regionall.save(f'{all}')
    cv2.imwrite(f'{leftlabel}', rotatelabel)
    regionup.save(f'{uplabel}')


# def line(org_img):
#     image = cv2.imread(f"{org_img}")
#     h = image.shape[0]
#     w = image.shape[1]
#
#     gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=1400, maxLineGap=40)
# lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=30)
# print(lines)

# liness = []
# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     if x1 == x2:
#         y1 = 0
#         y2 = h
#     elif y1 == y2:
#         x1 = 0
#         x2 = w
#     else:
#         continue
#     liness.append([x1, y1, x2, y2])
#     cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
# liness.sort(key=lambda x: (x[0], x[1]))
#
# latitude = []
# if liness[0][1] > 20:
#     latitude.append(liness[0][1])
# index = 1
# for i in range(len(liness)):
#     try:
#         height = liness[index][1] - liness[i][1]
#         if 20 < height:
#             latitude.append(liness[index][1])
#         if len(latitude) == 2:
#             break
#         index = index + 1
#     except:
#         continue
#
# liness.sort(key=lambda x: (x[2], x[3]))
# longitude = []
# index2 = 1
# if liness[0][2] > 20:
#     longitude.append(liness[0][2])
# for i in range(len(liness)):
#     try:
#
#         width = liness[index2][2] - liness[i][2]
#         if 20 < width:
#             longitude.append(liness[index2][2])
#         if len(longitude) == 2:
#             break
#         index2 = index2 + 1
#     except:
#         continue
# cv2.imwrite('./picdata/wen/line_detect_possible.jpg', image)
#
# return longitude, latitude

def tiqu(org_img, out_path):
    img = cv2.imread(f"{org_img}")
    # you can read in images with opencv
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # print(img_hsv)
    hsv_low = np.array([26, 43, 46])  # white!
    hsv_high = np.array([34, 255, 255])  # yellow! note the order
    print(hsv_high)
    # 对大于hsv_low和小于hsv_high的图像像素点均会被转化为0(黑色)，在此之间会转化为1(白色)
    mask = cv2.inRange(img_hsv, hsv_low, hsv_high)
    # print(mask)
    tiqu = out_path.joinpath('tiqu.png')
    cv2.imwrite(f'{tiqu}', mask)
    return tiqu


def ocr(org_img):
    # ssl._create_default_https_context = ssl._create_unverified_context
    # 创建reader对象
    reader = easyocr.Reader(['ch_sim', 'en'], model_storage_directory='./model/work/ocr',
                            download_enabled=False)
    # 读取图像
    result = reader.readtext(f'{org_img}', paragraph=True)
    img = cv2.imread(f"{org_img}")
    for box in result:
        x0, y0 = box[0][0][0], box[0][0][1]
        x2, y2 = box[0][2][0], box[0][2][1]
        cv2.rectangle(img, (int(x0), int(y0)), (int(x2), int(y2)), (0, 0, 0), -1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    gaus = cv2.GaussianBlur(gray, (19, 19), 0)

    edges = cv2.Canny(gaus, 50, 150, apertureSize=3)
    cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    #
    image_ori = cv2.imread(f"{org_img}")
    # cv2.drawContours(image_ori, cnts, -1,(255,0,0),1)
    # plt.imshow(image_ori)
    # plt.show()
    lines = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 2:
            X1, Y1, X2, Y2 = approx[0][0][0], approx[0][0][1], approx[1][0][0], approx[1][0][1]
            # cv2.line(image_ori, (X1, Y1), (X2, Y2), (0, 0, 255), 2)
            lines.append([[X1, Y1, X2, Y2]])
        else:
            kk = []
            for i in range(len(approx)):
                print(approx[i][0])
                kk.append(approx[i][0])
            approx = sorted(kk, key=lambda x: x[0])
            print(f'排序过的{approx}')
            dingdian = []
            duandian = [[0, 0]]
            print(approx[0])
            for i in approx:
                print(duandian[-1][0])
                if 10 > i[0] - duandian[-1][0] > -1:
                    dingdian.append(i)
                    del duandian[-1]
                elif i[0] - duandian[-1][0] > 10:
                    duandian.append(i)
                else:
                    continue
            del duandian[0]
            print(duandian)
            print(dingdian)
            for q in duandian:
                X1, Y1, X2, Y2 = q[0], q[1], dingdian[0][0], dingdian[0][1]
                # cv2.line(image_ori, (X1, Y1), (X2, Y2), (0, 0, 255), 2)
                lines.append([[X1, Y1, X2, Y2]]).imwrite('./dataset/newline.png', image_ori)
    print(lines)

    for i in range(len(result)):
        if '区' in result[i][1]:
            continue
        if '地' in result[i][1]:
            continue
        if '跑' in result[i][1]:
            continue
        if '道' in result[i][1]:
            continue
    text = []
    for i in range(len(result)):
        x1 = result[i][0][0][0]
        x2 = result[i][0][1][0]
        y1 = result[i][0][0][1]
        y4 = result[i][0][3][1]
        c = [int(((x2 - x1) / 2) + x1), int(((y4 - y1) / 2) + y1)]
        text.append([c, result[i][1]])
    return text, lines


polarLines = []
# 标签号集合
_index = []


# 给定线段的两个端点坐标，返回确定的直线的极坐标
def getPolarLine(p):
    # print(p[0])
    if (math.fabs(p[0] - p[2]) < 0.000001):
        if (p[0] > 0):
            return [p[0], 0]
        else:
            return [p[0], math.pi]
    elif (math.fabs(p[1] - p[3]) < 0.000001):
        if (p[1] > 0):
            return [p[1], math.pi / 2]
        else:
            return [p[1], 3 * math.pi / 2]
    else:
        k = (p[1] - p[3]) / (p[0] - p[2])
        y_intercept = p[1] - k * p[0]
        if (k < 0 and y_intercept > 0):
            theta = math.atan(-1 / k)
        elif (k > 0 and y_intercept > 0):
            theta = math.pi + math.atan(-1 / k)
        elif (k < 0 and y_intercept < 0):
            theta = math.pi + math.atan(-1 / k)
        elif (k > 0 and y_intercept < 0):
            theta = 2 * math.pi + math.atan(-1 / k)
        _cos = math.cos(theta)
        _sin = math.sin(theta)

        r = p[0] * _cos + p[1] * _sin

    return [r, theta]


def getIndexWithPolarLine(_index):
    polar_num = len(polarLines)
    if polar_num == 0:
        return False

    for i in range(0, polar_num):
        _index.insert(i, i)
    # print(_index)
    for j in range(0, polar_num - 1):
        minTheta = math.pi
        minR = 50
        polar1 = polarLines[j]
        for k in range(j + 1, polar_num):
            polar2 = polarLines[k]
            dTheta = math.fabs(polar2[1] - polar1[1])
            dR = math.fabs(polar2[0] - polar1[0])
            if dTheta < minTheta:
                minTheta = dTheta
            if dR < minR:
                minR = dR
            # 同类直线角度误差不超过1.8°，距离误差不超过8 %
            if dTheta < 1.8 * math.pi / 180 and dR < polar1[0] * 0.08:
                _index[k] = _index[j]
    return True


# def colorFilter(img_path):
#     img = cv2.imread(f'{img_path}')
#     gray_img=cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
#    # gaus = cv2.GaussianBlur(gray_img, (3, 3), 0)
#     edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
#
#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=7)
#      return lines

def getPositionPoints(lines, c):
    result = []
    total = []
    for line in lines:
        # print(line)
        polarLines.append(getPolarLine((line[0])))
    # print(polarLines)
    getIndexWithPolarLine(_index)
    # print(_index)
    _indexset = set(_index)
    # print(_indexset)
    _indexlines = []
    for iset in _indexset:
        _lines = []
        for i in range(0, len(_index)):
            if iset == _index[i]:
                x1, y1, x2, y2 = lines[i][0]
                _lines.append([x1, y1, x2, y2])
        _indexlines.append(_lines)
    # print(_indexlines)
    for i in range(0, len(_indexset)):
        MinX = _indexlines[i][0][0]
        MinY = _indexlines[i][0][1]
        MaxX = _indexlines[i][0][2]
        MaxY = _indexlines[i][0][3]
        for il in _indexlines[i]:
            if il[0] < MinX:
                MinX = il[0]
                MinY = il[1]
            if il[2] > MaxX:
                MaxX = il[2]
                MaxY = il[3]
        # print(MinX,MinY,MaxX,MaxY)
        result.append([MinX, MinY])
        result.append([MaxX, MaxY])
        total.append([[MinX, MinY], [MaxX, MaxY]])
    aa = []
    # zuobiao = [[0,500],[0,0],[500,0],[3000,500]]
    for k in c:
        print(k)
        count = [result[0]]
        for i in result:
            # print(f'原始坐标{count}')
            # print(f'原始点的个数{len(result)}')
            for j in range(len(count)):
                # print(f'数组{count}')
                # print(f'第一个数组{count[j]}')
                # print(f'新增距离为{distance(i,k)}，原始距离{distance(count[j], k)}')
                if distance(i, k) < distance(count[j], k):
                    count[j] = i
                else:
                    break
        aa.append(count[0])
        # print(f'输出列表为{count}')
    # print(f'原始列表{total}')
    # print(f'真正的输出列表为{aa}')

    new_result = []
    for p in aa:
        points = []
        for l in total:
            if p in l:
                c = l.copy()
                c.remove(p)
                # print(c)
                points.append(c[0])
            # print(points)
        new_result.append(points)
    logger.debug(f'输出的端点{new_result}')
    polarLines.clear()
    _index.clear()
    return new_result


def distance(point0, point1):
    v1 = np.array(point0) - np.array(point1)
    distance = np.sqrt(np.sum(v1 * v1))
    return distance


def touying(orgimg):
    img = cv2.imread(f'{orgimg}')
    GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将BGR图转为灰度图
    ret, thresh1 = cv2.threshold(GrayImage, 130, 255, cv2.THRESH_BINARY)  # 将图片进行二值化（130,255）之间的点均变为255（背景）
    # print(thresh1[0,0])#250 输出[0,0]这个点的像素值 				#返回值ret为阈值
    (h, w) = thresh1.shape  # 返回高和宽
    # print(h,w)#s输出高和宽
    a = [0 for z in range(0, w)]
    # print(a)  # a = [0,0,0,0,0,0,0,0,0,0,...,0,0]初始化一个长度为w的数组，用于记录每一列的黑点个数

    # 记录每一列的波峰
    for j in range(0, w):  # 遍历一列
        for i in range(0, h):  # 遍历一行
            if thresh1[i, j] == 0:  # 如果改点为黑点
                a[j] += 1  # 该列的计数器加一计数
                thresh1[i, j] = 255  # 记录完后将其变为白色
        # print(j)
    b = [0]
    #
    for j in range(0, w):  # 遍历每一列
        for i in range((h - a[j]), h):  # 从该列应该变黑的最顶部的点开始向最底部涂黑
            if 10 > j - b[-1] > -1:
                b[-1] = j
                continue
            if len(b) == 3:
                continue
            else:
                b.append(j)
            thresh1[i, j] = 0  # 涂黑
    b.remove(0)
    return b


def get_rotated_rect(image_path, points):
    # 定义在图像上绘制掩码的最小外接旋转矩形框的函数
    def draw_rotated_rect(mask):
        mask_8bit = (mask * 255).astype(np.uint8)
        _, binary_mask = cv2.threshold(mask_8bit, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(max_contour)
        return rect, max_contour

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sam_checkpoint = "./model/SAM/mobile_sam.pt"
    model_type = "vit_t"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam.eval()

    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)

    input_point = np.array(points)
    input_label = np.ones(len(points), dtype=np.int32)
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    rect, max_contour = draw_rotated_rect(masks[0])

    box_center, box_size, theta = rect

    h, w, _ = image.shape

    # 将max_contour转换为所需的格式
    contour_points = []
    for point in max_contour:
        x, y = point[0]
        contour_points.append([x / w, y / h])
    result = {
        "x": box_center[0] / w,
        "y": box_center[1] / h,
        "w": box_size[0] / w,
        "h": box_size[1] / h,
        "theta": str(int(theta)),
        "contour_points": contour_points
    }

    return result


def rotated_rect_to_bounding_box(cx, cy, width, height, angle):
    # 创建旋转矩形的四个顶点
    rect = ((cx, cy), (width, height), angle)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # 计算最小外接正矩形的四个顶点
    x, y, w, h = cv2.boundingRect(box)
    return x, y, x + w, y + h


def convert_to_pixel_coords(area, img_width, img_height):
    regions = []
    for region in area:
        x1 = int(region['left'] * img_width)
        y1 = int(region['top'] * img_height)
        x2 = x1 + int(region['width'] * img_width)
        y2 = y1 + int(region['height'] * img_height)
        regions.append((x1, y1, x2, y2))
    return regions


def is_in_region(box, region):
    # 简单示例：检查文本框中心点是否在矩形区域内
    box_center = ((box[0][0] + box[2][0]) / 2, (box[0][1] + box[2][1]) / 2)
    return (region[0] < box_center[0] < region[2]) and (region[1] < box_center[1] < region[3])


def calculate_similarity(text1, text2):
    """计算两个字符串的相似度，忽略空格和标点符号"""
    import re
    text1 = re.sub(r'\W+', '', text1)
    text2 = re.sub(r'\W+', '', text2)
    return sum(char1 == char2 for char1, char2 in zip(text1, text2)) / max(len(text1), len(text2))


def detect_and_recognize(image_path, regions):
    # 创建PaddleOCR实例
    ocr = PaddleOCR(lang="ch", ocr_version='PP-OCRv4', det_limit_side_len=840, cpu_threads=2, show_log=False)

    # 加载图像
    image = cv2.imread(image_path)
    img_height, img_width = image.shape[:2]

    # 定义两个感兴趣的矩形区域 (x1, y1, x2, y2)

    # 执行文本检测
    detected_boxes = ocr.ocr(image, det=True, rec=False)

    selected_boxes = [box for box in detected_boxes[0] for region in regions if is_in_region(box, region)]

    file_path = './area/true.txt'
    if os.path.exists(file_path):
        # 如果文件存在，删除它
        os.remove(file_path)

    with open(file_path, 'w') as file:

        for box in selected_boxes:
            # 裁剪文本框区域
            vertices = box
            # print(vertices)
            x_min = int(min([vertex[0] for vertex in vertices]))
            x_max = int(max([vertex[0] for vertex in vertices]))
            y_min = int(min([vertex[1] for vertex in vertices]))
            y_max = int(max([vertex[1] for vertex in vertices]))

            # 裁剪图像
            cropped_image = image[y_min:y_max, x_min:x_max]

            # 对裁剪的图像进行文本识别
            rec_result = ocr.ocr(cropped_image, det=False, rec=True)
            if rec_result:
                text = rec_result[0][0][0]
                print(text)
                box_percentage = {
                    "l_x": x_min / img_width,
                    "l_y": y_min / img_height,
                    "l_w": (x_max - x_min) / img_width,
                    "l_h": (y_max - y_min) / img_height
                }
                file.write(
                    f"{text}, {box_percentage['l_x']}, {box_percentage['l_y']}, {box_percentage['l_w']}, {box_percentage['l_h']}\n")


def parse_labels(img_path, file_path):
    labels = []
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    with open(file_path, 'r') as file:
        for index, line in enumerate(file):
            parts = line.strip().split(',')
            if len(parts) == 5:
                label = {
                    "order": index,
                    "class_id": "正确",
                    "l_x": float(parts[1].strip()),
                    "l_y": float(parts[2].strip()),
                    "l_w": float(parts[3].strip()),
                    "l_h": float(parts[4].strip()),
                    "w": w,
                    "h": h
                }
                labels.append(label)
    return labels


def str_to_bool(s):
    """将字符串转换为布尔值。"""
    if s.lower() in ['true', '1', 'yes', 'y', 't']:
        return True
    elif s.lower() in ['false', '0', 'no', 'n', 'f']:
        return False


def analyze_edge_colors(image, rect):
    x, y, w, h = rect
    # 确保不越界
    x_end = min(x + w, image.shape[1] - 1)
    y_end = min(y + h, image.shape[0] - 1)

    # 提取边缘像素
    top_edge = image[y, x:x_end]
    bottom_edge = image[y_end, x:x_end]
    left_edge = image[y:y_end, x]
    right_edge = image[y:y_end, x_end]

    edges = np.concatenate((top_edge, bottom_edge, left_edge, right_edge))

    # 计算颜色均值和标准差
    mean_color = np.mean(edges, axis=0)
    std_color = np.std(edges, axis=0)
    print(std_color)

    # 判断颜色是否一致（这里的阈值可以根据需要调整）
    if np.all(std_color < 30):  # 阈值为 10
        return True
    else:
        return False


def find_largest_black_rectangle_and_show_contours(image_path, reference_image_path):
    # 读取图像
    image = cv2.imread(image_path)
    image = resize_image(image, 480)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reference_image = cv2.imread(reference_image_path)
    reference_image = resize_image(reference_image, 480)

    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    # 应用高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 设置黑色阈值来提取黑色区域
    black_thresh = cv2.adaptiveThreshold(blurred, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)

    # 查找轮廓
    contours, _ = cv2.findContours(black_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 寻找最大的外接矩形
    largest_area = 0
    largest_rectangle = (0, 0, 0, 0)  # x, y, width, height
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > largest_area:
            largest_area = area
            largest_rectangle = (x, y, w, h)

    img_height, img_width = gray.shape[:2]
    img_area = img_width * img_height
    rectangle_area = largest_rectangle[2] * largest_rectangle[3]

    print(rectangle_area / img_area)

    # 检查最大矩形的面积是否在指定的范围内
    if rectangle_area > 0.9 * img_area or rectangle_area < 0.1 * img_area:
        client.write_registers(address=2, values=1, unit=1)
        client.write_registers(address=5, values=1, unit=1)
    else:
        client.write_registers(address=3, values=1, unit=1)
        client.write_registers(address=5, values=1, unit=1)
        return "未检测到标签"

    if analyze_edge_colors(image, largest_rectangle):
        client.write_registers(address=2, values=1, unit=1)
        client.write_registers(address=5, values=1, unit=1)
        print("标签处于标准背景中")
    else:
        client.write_registers(address=3, values=1, unit=1)
        client.write_registers(address=5, values=1, unit=1)
        return "标签可能不在标准背景中"
    #
    #
    # # 提取最大矩形区域
    # x, y, w, h = largest_rectangle  # 从前面的函数中获取
    # cropped_region = gray[y:y+h, x:x+w]
    #
    # sift = cv2.SIFT_create()
    #
    # # 检测并计算描述符
    # keypoints1, descriptors1 = sift.detectAndCompute(cropped_region, None)
    # keypoints2, descriptors2 = sift.detectAndCompute(reference_gray, None)
    #
    # # 创建匹配器并进行匹配
    # matcher = cv2.BFMatcher()
    # matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    #
    # # 应用比率测试
    # good_matches = []
    # for m, n in matches:
    #     if m.distance < 0.75 * n.distance:
    #         good_matches.append(m)
    #
    # # 绘制匹配结果
    # print(len(good_matches))
    #
    # if len(good_matches) < 40:
    #     return "标签大面积缺失或错误"
    # else:
    #     return "正确"


def resize_image(image, max_side_length):
    height, width = image.shape[:2]
    scale = max_side_length / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(image, (new_width, new_height))


def correction(image_path, width, height):
    image = cv2.imread(image_path)
    # print(image_path)
    target_height = 1500

    # 计算缩放比例
    scale_ratio = target_height / image.shape[0]

    # 缩放图像
    image = cv2.resize(image, None, fx=scale_ratio, fy=scale_ratio)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 设置黑色阈值来提取黑色区域
    black_thresh = cv2.adaptiveThreshold(blurred, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)

    # 查找轮廓
    contours, _ = cv2.findContours(black_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imwrite("./test/2.jpg", black_thresh)
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
    cv2.imwrite("./test/2.jpg", contour_image)

    # 检查第一大和第三大轮廓的面积差是否超过百分之50
    if len(sorted_contours) >= 3 and (one - three) / one > 0.5:
        max_contour = sorted_contours[0]
    else:
        max_contour = sorted_contours[2]

    total_area = image.shape[0] * image.shape[1]
    max_contour_area = cv2.contourArea(max_contour)
    print(max_contour_area)
    print(image.shape[0], image.shape[1], total_area)
    # 计算 max_contour 占原始图像面积的比例
    ratio = max_contour_area / total_area
    print(ratio)

    # import pdb;pdb.set_trace()

    # # 绘制最大轮廓
    # cv2.drawContours(image, [max_contour], -1, (0, 0, 255), 2)

    # 计算透视变换矩阵
    perspective_matrix, width, height, top_left = get_perspective_transform(image, max_contour, 1038, 1162)

    # 应用透视变换
    warped_image = cv2.warpPerspective(image, perspective_matrix, (1038, 1162))

    image_name = f"{uuid.uuid4()}.jpg"

    new_image_path = os.path.join("./correct", image_name)

    cv2.imwrite(new_image_path, warped_image)

    return new_image_path, ratio


def correction2(img):
    image = img

    # 计算缩放比例
    scale_ratio = 1

    # 缩放图像
    image = cv2.resize(image, None, fx=scale_ratio, fy=scale_ratio)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 设置黑色阈值来提取黑色区域
    black_thresh = cv2.adaptiveThreshold(blurred, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)

    # 查找轮廓
    contours, _ = cv2.findContours(black_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imwrite("./test/2.jpg", black_thresh)
    # areas = [cv2.contourArea(contour) for contour in contours]
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
    # print(f"轮廓数量{len(sorted_contours)}")

    # one = cv2.contourArea(sorted_contours[0])
    # three = cv2.contourArea(sorted_contours[2])
    # # print(cv2.contourArea(sorted_contours[0]))
    # # print(cv2.contourArea(sorted_contours[1]))
    # # print(cv2.contourArea(sorted_contours[2]))

    # 创建一个全白的空白图像
    contour_image = np.ones_like(image) * 255

    # 在空白图像上绘制第一大和第二大面积的边框
    cv2.drawContours(contour_image, sorted_contours, -1, (0, 0, 255), 2)

    # 保存绘制了边框的图像
    cv2.imwrite("./test/2.jpg", contour_image)

    # 检查第一大和第三大轮廓的面积差是否超过百分之50
    max_contour = sorted_contours[0]

    total_area = image.shape[0] * image.shape[1]
    max_contour_area = cv2.contourArea(max_contour)
    print(max_contour_area)
    print(image.shape[0], image.shape[1], total_area)
    # 计算 max_contour 占原始图像面积的比例
    ratio = max_contour_area / total_area
    print(ratio)

    # import pdb;pdb.set_trace()

    # # 绘制最大轮廓
    # cv2.drawContours(image, [max_contour], -1, (0, 0, 255), 2)

    # 计算透视变换矩阵
    perspective_matrix, width, height, top_left = get_perspective_transform(image, max_contour, 729, 150)

    # 应用透视变换
    warped_image = cv2.warpPerspective(image, perspective_matrix, (729, 150))

    image_name = f"{uuid.uuid4()}.jpg"

    new_image_path = os.path.join("./correct", image_name)

    cv2.imwrite(new_image_path, warped_image)

    return new_image_path, ratio, width, height, top_left


def get_perspective_transform(image, target_contour, width, height):
    # 提取目标区域的四个角点
    points = cv2.approxPolyDP(target_contour, 0.02 * cv2.arcLength(target_contour, True), True)

    # 找到与原图左上角最近的点的坐标
    nearest_to_origin = min(points, key=lambda point: np.linalg.norm(point[0]))

    # 找到与原图右上角最近的点的坐标
    nearest_to_top_right = min(points, key=lambda point: np.linalg.norm(point[0] - np.array([image.shape[1], 0])))

    # 找到与原图右下角最近的点的坐标
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

    dst_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    print(dst_points)
    # 计算透视变换矩阵
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    return perspective_matrix, width, height, top_left


def selfssim(img1, img2, folder1, folder2, smallthreshold=0.6, bigthreshold=0.58):
    image1 = cv2.imread(img1)
    image2 = cv2.imread(img2)

    # 将图像调整为相同的尺寸
    height, width = min(image1.shape[0], image2.shape[0]), min(image1.shape[1], image2.shape[1])
    image1 = cv2.resize(image1, (width, height))
    image2 = cv2.resize(image2, (width, height))

    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    #
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    ssim_score = ssim(gray_image1, gray_image2)
    print(f"第一次{ssim_score}")
    if ssim_score > bigthreshold:
        return True
    else:
        # 获取文件夹中的文件列表
        files1 = sorted(os.listdir(folder1))
        files2 = sorted(os.listdir(folder2))

        # 提取文件名中的后两位数字
        digits1 = [int(file[-6:-4]) for file in files1]
        digits2 = [int(file[-6:-4]) for file in files2]

        # 将文件按照后两位数字排序
        sorted_files1 = [x for _, x in sorted(zip(digits1, files1))]
        sorted_files2 = [x for _, x in sorted(zip(digits2, files2))]
        index = []
        # 逐一计算对应文件的 SSIM
        for idx, (file1, file2) in enumerate(zip(sorted_files1, sorted_files2)):
            file1_path = os.path.join(folder1, file1)
            # print(file1_path)
            file2_path = os.path.join(folder2, file2)
            # print(file2_path)
            image1 = cv2.imread(file1_path)
            image2 = cv2.imread(file2_path)

            # 将图像调整为相同的尺寸
            height, width = min(image1.shape[0], image2.shape[0]), min(image1.shape[1], image2.shape[1])
            image1 = cv2.resize(image1, (width, height))
            image2 = cv2.resize(image2, (width, height))

            gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

            ssim_score = ssim(gray_image1, gray_image2)
            # print(f"SSIM {idx}: {ssim_score}")
            if ssim_score < smallthreshold:
                # print(idx)
                # print(ssim_score)
                index.append(idx)
        print(f'集合{index}')
        if index == []:
            print("为空")
            return 1
        else:
            print("不为空")
            return index


def calcsim(img1, img2):
    for model_layer in resnet152v2.layers:
        model_layer.trainable = False

    def load_image(image_path):
        input_image = Image.open(image_path)
        resized_image = input_image.resize((640, 640))
        return resized_image

    def get_image_embeddings(object_image):
        image_array = np.expand_dims(image.img_to_array(object_image), axis=0)
        image_array = preprocess_input(image_array)
        image_embedding = resnet152v2.predict(image_array)
        return image_embedding

    def get_similarity_score(first_image, second_image):
        first_image = load_image(first_image)
        second_image = load_image(second_image)
        first_image_vector = get_image_embeddings(first_image)
        second_image_vector = get_image_embeddings(second_image)
        similarity_score = cosine_similarity(first_image_vector, second_image_vector).reshape(1, )
        return similarity_score

    similarity_score = get_similarity_score(img1, img2)
    return similarity_score


def matchdetect(query_imagepath, input_folder, output_folder):
    count = 0
    filename1 = Path(query_imagepath).stem
    query_image = cv2.imread(query_imagepath, cv2.IMREAD_GRAYSCALE)
    matched_image = cv2.cvtColor(query_image, cv2.COLOR_GRAY2BGR)
    original_height, original_width = query_image.shape

    for template_filename in os.listdir(input_folder):
        if template_filename.endswith((".jpg", ".jpeg", ".png")):
            template_image_path = os.path.join(input_folder, template_filename)
            print("Template Image Path:", template_image_path)

            # Read template image
            template_image = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)

            # Parse x, y from filename
            x, y, _ = map(int, template_filename.split('.')[0].split('-'))
            print(x, y)

            # Define template region
            template_height, template_width = template_image.shape

            max_similarity = -np.inf
            max_position = None

            # Define initial bounding box coordinates
            bbox_top_left = (max(0, x - 0.5 * template_width - 30), max(0, y - 0.5 * template_height - 30))
            bbox_bottom_right = (
                min(original_width, x + 0.5 * template_width + 30),
                min(original_height, y + 0.5 * template_height + 30))
            print(bbox_top_left, bbox_bottom_right)

            search_region = query_image[int(bbox_top_left[1]):int(bbox_bottom_right[1]),
                            int(bbox_top_left[0]):int(bbox_bottom_right[0])]

            for yy in range(int(bbox_top_left[1]), int(bbox_bottom_right[1]) - template_height):
                for xx in range(int(bbox_top_left[0]), int(bbox_bottom_right[0]) - template_width):
                    # Define region of interest (ROI)
                    roi = search_region[yy - int(bbox_top_left[1]):yy - int(bbox_top_left[1]) + template_height,
                          xx - int(bbox_top_left[0]):xx - int(bbox_top_left[0]) + template_width]

                    # Compute SSIM between template and ROI
                    similarity = cv2.matchTemplate(template_image, roi, cv2.TM_CCOEFF_NORMED)

                    # Get maximum similarity score and position
                    _, max_val, _, max_loc = cv2.minMaxLoc(similarity)

                    # Update maximum similarity score and position if needed
                    if max_val > max_similarity:
                        max_similarity = max_val
                        max_position = (xx, yy)

            print("Max Similarity:", max_similarity)
            if max_similarity < 0.6:
                count = count + 1
                # print("Max Position:", max_position)

                # Write maximum similarity score and position to file
                # Draw rectangle around the matched region
                cv2.rectangle(matched_image, max_position,
                              (max_position[0] + template_width, max_position[1] + template_height), (0, 0, 255), 2)
            # Write similarity score inside the rectangle
            cv2.putText(matched_image, f"{Path(template_filename).stem[-2:]}: {max_similarity:.2f}",
                        (max_position[0], max_position[1] + template_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 0, 0), 2)

    # Save matched image with rectangle and similarity score
    output_image_path = os.path.join(output_folder, f"{filename1}_matched.jpg")
    cv2.imwrite(output_image_path, matched_image)
    if count == 0:
        return count, output_image_path
    else:
        return count, output_image_path


def ocrimage(img_path):
    print(img_path)
    result = OCR.ocr(img_path, cls=True)
    rects = []
    texts = []
    # 遍历OCR识别结果和存储文本进行对比
    for line in result:
        # ocr_text = ''.join(c for c in line[1][0] if c.isalnum())
        rects.append([int(float(line[0][0][0])), int(float(line[0][0][1])), int(float(line[0][2][0])),
                      int(float(line[0][2][1]))])
        text = ''.join(c for c in line[1][0] if c.isalnum())
        texts.append(text)  # 提取文本信息
    return rects, texts


def process_image(image_path, results):
    image = cv2.imread(image_path)
    result = ocr.ocr(image, cls=True)
    results[image_path] = result  # 将结果与图像路径和索引关联


def ocrdetect(img_path, imgthreshold=0.8, graphthreshold=3, bottom_path=''):
    print(imgthreshold)
    print(graphthreshold)
    count = 0
    # results = {}
    # threads = []
    # new_bottom_path = cv2.imread(bottom_path)
    # for i, image_path in enumerate([img_path, correction2(new_bottom_path)]):
    #     thread = threading.Thread(target=process_image, args=(image_path, results))
    #     threads.append(thread)
    #     thread.start()
    # for thread in threads:
    #     thread.join()
    # result = results[img_path]
    # resultB = results[new_bottom_path]
    result = OCR.ocr(img_path, cls=True)[0]
    resultB = None
    if bottom_path != '':
        new_image_path, ratio, width, height, top_left = correction2(cv2.imread(bottom_path))
        resultB = OCR.ocr(new_image_path, cls=True)[0]

    # 读取图像
    img = cv2.imread(img_path)
    imgB = cv2.imread(bottom_path)
    original_height, original_width, channels = img.shape

    original_heightB, original_widthB, channelsB = imgB.shape

    # 列出文件夹中的所有文件
    files = [file for file in os.listdir("./industrialLabels/sample/subZoneFolder/") if
             file.endswith('.png')]

    # 遍历文件
    for file in files:
        filename = Path(file).stem
        file_parts = filename.split("-")
        last_part = file_parts[-1]
        print(last_part)
        # 检查倒数第二个字符是否为9
        if int(last_part) > 90:
            # 构建完整路径
            template_image_path = os.path.join("./industrialLabels/sample/subZoneFolder/", file)
            # 打印路径或者执行你需要的操作
            template_filename = str(Path(template_image_path).name)
            # Read template image
            template_image = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)
            search_region_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Parse x, y from filename
            x, y, _ = map(int, template_filename.split('.')[0].split('-'))
            print(x, y)

            # Define template region
            template_height, template_width = template_image.shape

            max_similarity = -np.inf
            max_position = None

            # Define initial bounding box coordinates
            bbox_top_left = (max(0, x - 0.5 * template_width - 5), max(0, y - 0.5 * template_height - 5))
            bbox_bottom_right = (min(original_width, x + 0.5 * template_width + 5),
                                 min(original_height, y + 0.5 * template_height + 5))
            print(bbox_top_left, bbox_bottom_right)

            search_region = search_region_gray[int(bbox_top_left[1]):int(bbox_bottom_right[1]),
                            int(bbox_top_left[0]):int(bbox_bottom_right[0])]

            for yy in range(int(bbox_top_left[1]), int(bbox_bottom_right[1]) - template_height):
                for xx in range(int(bbox_top_left[0]), int(bbox_bottom_right[0]) - template_width):
                    # Define region of interest (ROI)
                    roi = search_region[yy - int(bbox_top_left[1]):yy - int(bbox_top_left[1]) + template_height,
                          xx - int(bbox_top_left[0]):xx - int(bbox_top_left[0]) + template_width]

                    # Compute SSIM between template and ROI
                    similarity = cv2.matchTemplate(template_image, roi, cv2.TM_CCOEFF_NORMED)

                    # Get maximum similarity score and position
                    _, max_val, _, max_loc = cv2.minMaxLoc(similarity)

                    # Update maximum similarity score and position if needed
                    if max_val > max_similarity:
                        max_similarity = max_val
                        max_position = (xx, yy)
            print("Max Similarity:", max_similarity)

            # print("Max Similarity:", max_similarity)
            if 0 < max_similarity < imgthreshold:
                print("Max Similarity:", max_similarity)

                print("画框")
                # print("Max Position:", max_position)
                count = count + 1

                # Write maximum similarity score and position to file
                # Draw rectangle around the matched region
                cv2.rectangle(img, max_position,
                              (max_position[0] + template_width, max_position[1] + template_height), (0, 0, 255), 2)
    # OCR识别

    # 读取存储文本文件

    stored_rects = []
    stored_texts = []
    stored_rects_texts = []
    rects_tests = {}

    with open('./ocrsample/ocr_result.txt', 'r', encoding='utf-8') as file:
        for line in file:
            # 分割每行并转换为相应的数据格式
            line_split = line.strip().split(' ')
            stored_rect = [int(float(x)) for x in line_split[:4]]  # 前4个数是矩形坐标
            stored_text = ''.join(line_split[4:])  # 提取文本内容

            x1, y1, x2, y2 = stored_rect
            x = x1
            y = y1
            w = x2 - x1
            h = y2 - y1
            stored_rect = x, y, w, h
            # 存储矩形和文本
            stored_rects.append(stored_rect)
            stored_texts.append(stored_text)
            stored_rects_texts.append([stored_rect, stored_text])
            rects_tests[stored_rect] = stored_text

    # print(stored_texts)

    # 计算两个矩形的交并比（IoU）
    def calculate_iou(rect1, rect2):
        # 提取矩形的坐标
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2

        # 计算交集部分的坐标
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        # 计算交集面积
        intersection_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)

        # 计算并集面积
        rect1_area = (w1 + 1) * (h1 + 1)
        rect2_area = (w2 + 1) * (h2 + 1)
        union_area = rect1_area + rect2_area - intersection_area

        # 计算交并比
        iou = intersection_area / union_area

        return iou

    outputpath = ""
    rects = []
    texts = []

    rectsB = []
    textsB = []

    # 遍历OCR识别结果和存储文本进行对比
    for line in result:
        # ocr_text = ''.join(c for c in line[1][0] if c.isalnum())
        rects.append([int(float(line[0][0][0])), int(float(line[0][0][1])), int(float(line[0][2][0])),
                      int(float(line[0][2][1]))])
        text = ''.join(c for c in line[1][0] if c.isalnum())
        texts.append(text)  # 提取文本信息

    connector = BoxesConnector(rects, texts, original_width, max_dist=20, overlap_threshold=0.4)
    new_rects, new_texts = connector.connect_boxes()
    # 遍历OCR识别结果和存储文本进行对比

    if bottom_path != '':
        # 小图片
        stored_rectsB = []
        stored_textsB = []
        stored_rects_textsB = []
        # 在全图上的位置
        real_stored_rectsB = []
        real_stored_rects_textsB = []
        with open('./ocrsample/ocr_bottom_result.txt', 'r', encoding='utf-8') as file:
            for line in file:
                # 分割每行并转换为相应的数据格式
                line_split = line.strip().split(' ')
                stored_rect = [int(float(x)) for x in line_split[:4]]  # h4个数是矩形坐标 小坐标
                stored_text = ''.join(line_split[4])  # 提取文本内容

                x1, y1, x2, y2 = stored_rect
                x = x1
                y = y1
                w = x2 - x1
                h = y2 - y1
                stored_rect = x, y, w, h
                # 存储矩形和文本
                stored_rectsB.append(stored_rect)
                stored_textsB.append(stored_text)
                stored_rects_textsB.append([stored_rect, stored_text])

                stored_rect = [int(float(x)) for x in line_split[-4:]]  # 前4个数是矩形坐标 大坐标

                x1, y1, x2, y2 = stored_rect
                x = x1
                y = y1
                w = x2 - x1
                h = y2 - y1
                stored_rect = x, y, w, h
                # 存储矩形和文本
                real_stored_rectsB.append(stored_rect)
                real_stored_rects_textsB.append([stored_rect, stored_text])
        for line in resultB:
            # ocr_text = ''.join(c for c in line[1][0] if c.isalnum())
            rectsB.append([int(float(line[0][0][0])), int(float(line[0][0][1])), int(float(line[0][2][0])),
                           int(float(line[0][2][1]))])
            text = ''.join(c for c in line[1][0] if c.isalnum())
            textsB.append(text)  # 提取文本信息

        connector = BoxesConnector(rectsB, textsB, original_widthB, max_dist=20, overlap_threshold=0.4)
        new_rectsB, new_textsB = connector.connect_boxes()

        #
        for ocr_rect, ocr_text in new_textsB.items():
            x1, y1, x2, y2 = ocr_rect
            x = x1
            y = y1
            w = x2 - x1
            h = y2 - y1
            ocr_rect = x, y, w, h
            # 获取OCR矩形框的坐标

            # 初始化最大IoU值和对应的存储文本索引
            max_iou = -1
            max_iou_index = -1

            # 遍历所有存储的矩形框
            for i, stored_rect in enumerate(stored_rectsB):
                # 计算当前OCR矩形框与存储的矩形框的IoU值
                iou = calculate_iou(ocr_rect, stored_rect)

                # 如果当前IoU值大于最大IoU值，则更新最大IoU值和对应的存储文本索引
                if iou > max_iou:
                    max_iou = iou
                    max_iou_index = i
            # 如果最大IoU值大于等于阈值
            if max_iou >= 0.1:
                stored_text_matched = stored_textsB[max_iou_index]  # 匹配到的文字清晰文字
                stored_rects_matched = real_stored_rectsB[max_iou_index]  # 匹配到的文字在原图上的坐标
                # print("匹配上了ocr_bottom_result.txt中的文字："+stored_textsB[max_iou_index] + "和识别到的文字坐标" + ocr_text,ocr_rect)
                # 标记为已匹配，匹配到之后将文字替换
                # 如何替换
                max_iouB = -1
                max_rect = ''
                max_text = ''
                for i, (rect, text) in enumerate(new_texts.items()):
                    x1, y1, x2, y2 = rect
                    x = x1
                    y = y1
                    w = x2 - x1
                    h = y2 - y1
                    ocr_rect = x, y, w, h
                    # 计算当前OCR矩形框与存储的矩形框的IoU值
                    iou = calculate_iou(stored_rects_matched, ocr_rect)
                    # 如果当前IoU值大于最大IoU值，则更新最大IoU值和对应的存储文本索引
                    if iou > max_iouB:
                        max_iouB = iou
                        max_rect = rect
                        max_text = text
                if max_iouB >= 0.1:
                    new_texts[max_rect] = ocr_text

    # 开始校验
    matched_rects = []
    for ocr_rect, ocr_text in new_texts.items():
        x1, y1, x2, y2 = ocr_rect
        x = x1
        y = y1
        w = x2 - x1
        h = y2 - y1
        ocr_rect = x, y, w, h
        # 获取OCR矩形框的坐标

        # 初始化最大IoU值和对应的存储文本索引
        max_iou = -1
        max_iou_index = -1

        # 遍历所有存储的矩形框
        for i, stored_rect in enumerate(stored_rects):
            # 计算当前OCR矩形框与存储的矩形框的IoU值
            iou = calculate_iou(ocr_rect, stored_rect)

            # 如果当前IoU值大于最大IoU值，则更新最大IoU值和对应的存储文本索引
            if iou > max_iou:
                max_iou = iou
                max_iou_index = i
        # print(ocr_text)

        # 如果最大IoU值大于等于阈值
        if max_iou >= 0.1:
            stored_text_matched = stored_texts[max_iou_index]
            # stored_rects[max_iou_index] = (None, None, None, None)  # 标记为已匹配
            matched_rects.append(stored_rects[max_iou_index])

            # 获取最大交并比对应的存储文本
            print(max_iou)

            print(f"识别的{ocr_text}")
            print(f"存储的{stored_text_matched}")
            if len(ocr_text) >= 10 and ocr_text.isdigit():
                if ocr_text == stored_text_matched:
                    print("跳过")
                    continue
                print("画框")
                count = count + 1
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)


            elif len(ocr_text) >= 6:

                if re.search(
                        r'[A-Za-z]{5,}.*[\u4e00-\u9fa5]+|[\u4e00-\u9fa5]+.*[A-Za-z]{5,}', ocr_text) or re.search(
                    r'[A-Za-z]{2,}.*\d{3,}|\d{3,}.*[A-Za-z]{2,}', ocr_text):
                    print(ocr_text)
                    print(stored_text_matched)
                    if len(ocr_text) == len(stored_text_matched):
                        if ocr_text == stored_text_matched:
                            print("跳过")
                            continue
                        difference = [char for char in ocr_text if char not in stored_text_matched] + [char for char in
                                                                                                       stored_text_matched
                                                                                                       if
                                                                                                       char not in ocr_text]
                        difference_string = ''.join(difference)
                        if difference_string in ['O', '0']:
                            print(difference_string)

                            continue
                    print("文本画框")

                    count = count + 1
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # 如果OCR识别结果和存储的文本内容不完全一样
            if abs(len(ocr_text) - len(stored_text_matched)) <= graphthreshold:
                print(ocr_text)
                print(stored_text_matched)
                print("跳过")
                continue
            print("画框")

            count = count + 1
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # 遍历所有的stored_rects，找出未被匹配到的矩形框，并将其画出来
    print(f"匹配到的{matched_rects}")
    # print(stored_rects)
    # # 绘制未匹配到的矩形框
    # print(len(matched_rects))
    # print(len(stored_rects))
    for stored_rect in stored_rects_texts:
        if stored_rect[0] not in matched_rects:
            x, y, w, h = stored_rect[0]
            print(len(stored_rect[1]))
            if len(stored_rect[1]) <= graphthreshold:
                print("跳过")
                continue
            count = count + 1
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            print(f"画出的{stored_rect[0]}")

    if count != 0:
        outputpath = f"./false/{Path(img_path).name}"
        cv2.imwrite(outputpath, img)

    return count, outputpath


# import os
# import shutil
# from shapely.geometry import Polygon
#
# def merge_boxes(box1, box2):
#     """Merge two bounding boxes."""
#     x1, y1, x2, y2 = box1
#     x3, y3, x4, y4 = box2
#     merged_box = [min(x1, x3), min(y1, y3), max(x2, x4), max(y2, y4)]
#     return merged_box
#
# def calculate_iou(box1, box2):
#     """Calculate intersection over union (IOU) of two bounding boxes."""
#     poly1 = Polygon([(box1[0], box1[1]), (box1[2], box1[3]), (box1[4], box1[5]), (box1[6], box1[7])])
#     poly2 = Polygon([(box2[0], box2[1]), (box2[2], box2[3]), (box2[4], box2[5]), (box2[6], box2[7])])
#     iou = poly1.intersection(poly2).area / poly1.union(poly2).area
#     return iou
#
# def merge_and_write_results(input_file, output_file, iou_threshold=0):
#     """Merge overlapping bounding boxes and write the results to a new file."""
#     with open(input_file, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#
#     merged_boxes = []
#     merged_texts = []
#
#     for line in lines:
#         # print(line)
#         points_str, text = line.strip().split(' ', 1)
#         print(points_str)
#         points = list(map(float, points_str.split()))
#         print(points)
#         # 解析坐标点
#         current_box = [(points[i], points[i+1]) for i in range(0, len(points), 2)]
#         print(current_box)
#             # Calculate IOU with previous box
#         iou = calculate_iou(current_box, merged_boxes[-1])
#         print(iou)
#         if iou > iou_threshold:
#             # Merge boxes and texts
#             merged_boxes[-1] = merge_boxes(current_box, merged_boxes[-1])
#             merged_texts[-1] += text
#             continue
#
#         # Append new box and text
#         merged_boxes.append(current_box)
#         merged_texts.append(text)
#
#     # Write merged results to new file
#     with open(output_file, 'w', encoding='utf-8') as f:
#         for box, text in zip(merged_boxes, merged_texts):
#             box_str = ' '.join(str(coord) for point in box for coord in point)
#             f.write(f"{box_str} {text}\n")
#
# # 调用OCR函数获取结果并写入文件
# def ocrsample(img_path):
#     result = OCR.ocr(img_path, cls=True)
#     folder_path = "./ocrsample"
#     output_file = os.path.join(folder_path, "merged_ocr_result.txt")
#
#     # 写入OCR结果到文件
#     with open(os.path.join(folder_path, "ocr_result.txt"), 'w', encoding='utf-8') as f:
#         for line in result:
#             # 获取文本内容并去除所有符号（包括标点符号和空格）
#             text = ''.join(c for c in line[1][0] if c.isalnum())
#             if "Water" in text or "NSF" in text or "An" in text:
#                 print(text)
#                 print("跳过")
#                 continue
#             # 获取文本框坐标并转换为字符串
#             box_str = ' '.join(str(coord) for point in line[0] for coord in point)
#             # 将结果写入到txt文件中
#             f.write(f"{box_str} {text}\n")
#
#     # 合并重叠的文本框并写入到新文件中
#     merge_and_write_results(os.path.join(folder_path, "ocr_result.txt"), output_file)

def ocrsample(img_path):
    folder_path = "./ocrsample"
    if os.path.exists(folder_path):
        # 清空文件夹
        shutil.rmtree(folder_path)
        # 重新创建空文件夹
        os.makedirs(folder_path)
    image = cv2.imread(img_path)
    # 底部ocr保存 #################################################################
    # 获取图像的高度和宽度
    height, width = image.shape[:2]

    # 计算下30%部分的起始行
    start_row = int(height * 0.5)

    # 提取下50%部分的图像
    bottom_50_percent_image = image[start_row:height, :]
    new_image_path, _, width, height, top_left = correction2(bottom_50_percent_image)
    top_left[1]=top_left[1]+start_row
    shutil.move(new_image_path, "./ocrsample/bottom.jpg")
    bottomImageResult = OCR.ocr("./ocrsample/bottom.jpg", cls=True)[0]

    with open('./ocrsample/ocr_bottom_result.txt', 'w', encoding='utf-8') as f:
        for line in bottomImageResult:
            text = ''.join(c for c in line[1][0] if c.isalnum())
            box_str = ' '.join(str(coord) for point in line[0] for coord in point)
            # 将结果写入到txt文件中
            f.write(f"{box_str} {text}\n")

    with open('./ocrsample/ocr_bottom_result.txt', 'r') as file:
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
    with open('./ocrsample/ocr_bottom_result.txt', 'w') as file:
        for rect, text in new_texts.items():
            # 将矩形框和文本以指定格式写入文件
            file.write(
                f"{rect[0]} {rect[1]} {rect[2]} {rect[3]} {text} {rect[0] + top_left[0]} {rect[1] + top_left[1]} {rect[2] + top_left[0]} {rect[3] + top_left[1]}\n")

    # OCR 识别整张图片 排除botton
    result = OCR.ocr(img_path, cls=True)[0]
    width = image.shape[1]
    # print(result)
    # 读取另外一个文本文件的前两行
    with open('./industrialLabels/sample/sampleTxt.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if len(lines) > 2:
            lines = [''.join(c for c in line if c.isalnum()) for line in lines[:2]]
            last_line = ''.join(c for c in lines[-1] if c.isalnum())
            lines.append(last_line)
        lines = [''.join(c for c in line if c.isalnum()) for line in lines[:2]]
        # print(lines)

    with open('./ocrsample/ocr_result.txt', 'w', encoding='utf-8') as f:
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

    with open('./ocrsample/ocr_result.txt', 'r') as file:
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
    with open('./ocrsample/ocr_result.txt', 'w') as file:
        for rect, text in new_texts.items():
            # 将矩形框和文本以指定格式写入文件
            file.write(f"{rect[0]} {rect[1]} {rect[2]} {rect[3]} {text}\n")


def levenshtein_distance(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).ratio()


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


# def calculate_iou(box1, box2):
#     # 计算两个矩形框的交集部分
#     x_left = max(box1[0], box2[0])
#     y_top = max(box1[1], box2[1])
#     x_right = min(box1[2], box2[2])
#     y_bottom = min(box1[3], box2[3])
#
#     # 计算交集面积
#     intersection_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)
#
#     # 计算两个矩形框的面积
#     box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
#     box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
#
#     # 计算IoU
#     iou = intersection_area / float(box1_area + box2_area - intersection_area)
#
#     return iou
def get_rect_points(text_boxes):
    x1 = np.min(text_boxes[:, 0])
    y1 = np.min(text_boxes[:, 1])
    x2 = np.max(text_boxes[:, 2])
    y2 = np.max(text_boxes[:, 3])
    return [x1, y1, x2, y2]


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

#
img1 = '/Users/du/Desktop/7-2/2024-07-02_14-49-50-sUpO2W.jpg'
img2 = '/Users/du/Desktop/7-2/2024-07-02_14-49-50-59tymQ.jpg'
correction_path, _ = correction(img1, "", "")
count, outputpath = ocrdetect(correction_path, 0.6, 1,
                              img2)
print(count, outputpath)

# ocrsample('industrialLabels/sample/sampleSplit.jpg')
