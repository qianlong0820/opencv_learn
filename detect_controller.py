import logging
import shutil
import threading
import time
from pathlib import Path
from PIL import Image
import subprocess
import torch
from app import config
from app.AudioCLIP.test import audioclassify
from app.NAFNet.basicsr.models import create_model
from app.NAFNet.basicsr.train import parse_options
from app.NAFNet.basicsr.utils import tensor2img, imwrite, img2tensor, imfrombytes, FileClient
from app.config import basepath
from app.model.Entity import Entity
from app.model.Indicia_entity import IndiciaEntity
from app.model.label_entity import LabelEntity
from app.model.respon_entity import ResponEntity
from app.service.detect_service import cutimage, detect_indicia, ocr, tiqu, getPositionPoints, touying, \
    get_rotated_rect, rotated_rect_to_bounding_box, \
    detect_and_recognize, str_to_bool, convert_to_pixel_coords, find_largest_black_rectangle_and_show_contours, \
    parse_labels, correction, selfssim, calcsim, matchdetect, ocrdetect, ocrsample, ocrimage
from detect import run
from gongpack import gong, imgsvc
# from pymodbus.client.sync import ModbusTcpClient
#
# # Modbus TCP服务器的IP地址和端口号
# server_ip = '192.168.0.11'
# server_port = 502
#
# # 创建一个Modbus TCP客户端
# client = ModbusTcpClient(server_ip, port=server_port)
#
# # 连接到Modbus设备
# client.connect()

import cv2
logger = logging.getLogger(config.app_name)
import falcon

import simplejson as json

class DetectController:
    def __init__(self):
        pass

class IndiciaCombineDetectController(DetectController):
    def on_get(self, req, resp):
        # 拆分后的图片路径列表
        originPath = req.params['originPath']
        if not Path(originPath).exists():
            raise BaseException
        outpath = Path(basepath).joinpath('picdata').joinpath('wen')
        if outpath.exists():
            shutil.rmtree(outpath)
        outpath.mkdir(parents=True)
        # logger.debug(f'{outpath}')
        try:
            cutimage(originPath, outpath)
            model_path = Path(basepath).joinpath('model').joinpath('ocr')
            leftpath = outpath.joinpath('left.png')
            uppath = outpath.joinpath('up.png')
            latitude = detect_indicia(f'{leftpath}', model_path)
            longitude = detect_indicia(f'{uppath}', model_path)
            logger.debug(f'{longitude}')
            logger.debug(f'{latitude}')
            allpath = outpath.joinpath('all.png')
            tiqupath = tiqu(allpath, outpath)
            uplabel = outpath.joinpath('leftlabel.png')
            leftlabel = outpath.joinpath('uplabel.png')
            latitudeimage = touying(f'{leftlabel}')
            longitudeimage = touying(f'{uplabel}')
            klong = (longitude[1] - longitude[0]) / (longitudeimage[1]-longitudeimage[0])
            klat = (latitude[1] -latitude[0]) / (latitudeimage[1]-latitudeimage[0])
            # logger.debug(f'纬度比例{klat}，经度比例{klong}')
            shuchu = ocr(tiqupath)
            # logger.debug(f'结果为{shuchu}')
            aa = shuchu[0]
            center = []
            for i in aa:
                center.append(i[0])
            tiqulines = shuchu[1]
            bb = getPositionPoints(tiqulines, center)
            indicias = []
            # logger.debug(f'初始经度{longitude[0]}')
            # logger.debug(f'初始纬度{latitude[0]}')
            index = 0
            for i in range(len(aa)):
                # print(c)
                for j in range(len(bb[i])):
                    indiciaEntity = IndiciaEntity()
                    indiciaEntity.order = index
                    indiciaEntity.content = aa[i][1]
                    indiciaEntity.x = (bb[i][j][0] - longitudeimage[0]) * klong + longitude[0] + 0.001
                    indiciaEntity.y = (2345 - bb[i][j][1] - latitudeimage[0]) * klat + latitude[0]
                    indiciaEntity.X = bb[i][j][0] + 455
                    indiciaEntity.Y = bb[i][j][1] + 530
                    index = index + 1
            # logger.debug(f'输出的数组{indicias}')z
            resp.body = json.dumps(ResponEntity().ok("识别标注成功", indicias))
            resp.status = falcon.HTTP_200
        except Exception as e:
            resp.body = json.dumps(ResponEntity().exception("识别标注失败", e))
            resp.status = falcon.HTTP_200


# OBB目标识别
class TargetDetectController(DetectController):
    def on_get(self, req, resp):
        source = str(req.params['imgPath'])
        weight = str(req.params['modelPath'])
        outpath = str(req.params['outPath'])
        if Path(outpath).exists():
            shutil.rmtree(outpath)
        Path(outpath).mkdir(parents=True)
        try:
            img = Image.open(source)
            hh, ww = img.size[1], img.size[0]
            labels = []
            dopt = Entity()
            dopt.agnostic_nms = False
            dopt.classes = None
            dopt.conf_thres = float(0.25)
            dopt.device = 'cpu'
            dopt.iou_thres = float(0.25)
            dopt.imgsz = (hh, ww)
            dopt.source = source
            dopt.weights = weight
            result = run(dopt)
            templabels = []

            # 生成的标签文件txt路径
            filename = Path(source).stem
            filepath = Path(outpath).joinpath(f'{filename}.txt')

            # 过滤靠的太近的目标框并保留最高的置信度
            for index, i in enumerate(result):
                linee = result[index].split(' ')
                # print(f'第一轮{line}')
                for le in templabels:
                    distance = pow(pow(float(linee[0]) - float(le[0]), 2) + pow(float(linee[1]) - float(le[1]), 2), 0.5)
                    if distance < 5:
                        if linee[6] < le[6]:
                            templabels.remove(linee)
                        else:
                            templabels.remove(le)
                        continue
                templabels.append(linee)
            tagtxt = ''
            for index, line in enumerate(templabels):
                tagline = f'{line[0]} {line[1]} {line[5]}'
                if tagtxt == '':
                    tagtxt = tagline
                else:
                    tagtxt = f'{tagtxt}\n{tagline}'
                labelEntity = LabelEntity()
                labelEntity.order = index
                labelEntity.class_id = line[5]
                labelEntity.probability = line[6]
                # 转换浮点数为整形
                x, y, w, h = round(float(line[0])) / ww, round(float(line[1])) / hh, round(float(line[2])) / ww, round(
                    float(line[3])) / hh
                labelEntity.l_x = x - w / 2
                labelEntity.l_y = y - h / 2
                labelEntity.r_x = x + w / 2
                labelEntity.r_y = y + h / 2
                labelEntity.theta = line[4]
                labels.append(labelEntity.obj2dct())
                # print(labels)
            with open(filepath, 'w') as f_out:
                f_out.write(tagtxt)
            resp.body = json.dumps(ResponEntity().ok("检测目标成功", labels))
            resp.status = falcon.HTTP_200
        except Exception as e:
            resp.body = json.dumps(ResponEntity().exception("检测目标失败", e))
            resp.status = falcon.HTTP_200



class NAFNetController(DetectController):
    def on_get(self, req, resp):
        # 拆分后的图片路径列表
        input_img = req.params['inputImg']
        opt = req.params['opt']
        if not Path(input_img).exists():
            raise BaseException
        # filename = Path(input_img).name
        output_img = str(Path('/home/ya/mapdata/linear').joinpath(f'{Path(input_img).stem}_{gong.dt2str()}.jpg'))

        try:
            opt = parse_options(input_img, output_img, opt, is_train=False)
            # print(opt)
            opt['num_gpu'] = torch.cuda.device_count()

            img_path = opt['img_path'].get('input_img')
            output_path = opt['img_path'].get('output_img')

            ## 1. read image
            file_client = FileClient('disk')

            img_bytes = file_client.get(img_path, None)
            try:
                img = imfrombytes(img_bytes, float32=True)
            except:
                raise Exception("path {} not working".format(img_path))

            img = img2tensor(img, bgr2rgb=True, float32=True)

            ## 2. run inference
            opt['dist'] = False
            model = create_model(opt)

            model.feed_data(data={'lq': img.unsqueeze(dim=0)})

            if model.opt['val'].get('grids', False):
                model.grids()

            model.test()

            if model.opt['val'].get('grids', False):
                model.grids_inverse()

            visuals = model.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            imwrite(sr_img, output_path)

            print(f'inference {img_path} .. finished. saved to {output_path}')

            # logger.debug(f'输出的数组{indicias}')z
            resp.body = json.dumps(ResponEntity().ok("图像智能处理成功", output_img))
            resp.status = falcon.HTTP_200
        except Exception as e:
            resp.body = json.dumps(ResponEntity().exception("图像智能处理失败", e))
            resp.status = falcon.HTTP_200




class AudioClassifyController(DetectController):
    def on_get(self, req, resp):
        # 拆分后的图片路径列表
        input_dir = req.params['inputDir']
        model_path = req.params['modelPath']
        if not Path(input_dir).exists():
            raise BaseException

        try:
            results = audioclassify(model_path, input_dir)

            # logger.debug(f'输出的数组{indicias}')z
            resp.body = json.dumps(ResponEntity().ok("音频分类成功", results))
            resp.status = falcon.HTTP_200
        except Exception as e:
            resp.body = json.dumps(ResponEntity().exception("音频分类失败", e))
            resp.status = falcon.HTTP_200




class AutoSegmentController(DetectController):
    def on_post(self, req, resp):
        path = req.media["picPath"]
        labels = req.media["points"]

        try:
            points = [list(map(int, label.split(','))) for label in labels]
            result = get_rotated_rect(path, points)

            resp.body = json.dumps(ResponEntity().ok(
                "提取目标成功", result
            ))
            resp.status = falcon.HTTP_200
        except Exception as e:
            resp.body = json.dumps(ResponEntity().exception("提取目标失败", e))
            resp.status = falcon.HTTP_200




class LabelDetectController(DetectController):
    def on_get(self, req, resp):
        source = str(req.params['imgPath'])
        try:
            img = Image.open(source)
            hh, ww = img.size[1], img.size[0]
            labels = []
            dopt = Entity()
            dopt.agnostic_nms = False
            dopt.classes = None
            dopt.conf_thres = float(0.25)
            dopt.device = 'cpu'
            dopt.iou_thres = float(0.25)
            dopt.imgsz = (720, 1280)
            dopt.source = source
            dopt.weights = '/home/ya/mapdata/model/work/best.pt'
            result = run(dopt)
            print(result)
            for i, data in enumerate(result):
                values = data.split()
                cx, cy, width, height, angle, class_id, probability = float(values[0]), float(values[1]), float(
                    values[2]), float(values[3]), float(values[4]), values[5], float(values[6])
                x1, y1, x2, y2 = rotated_rect_to_bounding_box(cx, cy, width, height, angle)

                # 转换为相对坐标和尺寸
                l_x = x1 / ww
                l_y = y1 / hh
                l_w = (x2 - x1) / ww
                l_h = (y2 - y1) / hh

                formatted_result = {
                    "order": i,
                    "class_id": class_id,
                    "l_x": l_x,
                    "l_y": l_y,
                    "l_w": l_w,
                    "l_h": l_h,
                    "w": ww,
                    "h": hh
                }
                labels.append(formatted_result)

            resp.body = json.dumps(ResponEntity().ok("检测目标成功", labels))
            resp.status = falcon.HTTP_200
        except Exception as e:
            resp.body = json.dumps(ResponEntity().exception("检测目标失败", e))
            resp.status = falcon.HTTP_200



class SaveLabelsController(DetectController):
    def on_post(self, req, resp):
        path = req.media["imgPath"]
        area = req.media["area"]

        try:
            # print(path)
            # print(area)
            image = cv2.imread(path)
            img_height, img_width = image.shape[:2]

            # regions = [
            #     (906, 228, 2096, 326),  # 第一个矩形区域的坐标
            #     (919, 315, 2121, 471),  # 第一个矩形区域的坐标
            #     (1378, 2456, 1522, 2481),
            #     (1712, 2459, 1857, 2485),
            #     (2106, 2434, 2298, 2472)
            # ]

            regions = convert_to_pixel_coords(area, img_width, img_height)

            detect_and_recognize(path, regions)


            resp.body = json.dumps(ResponEntity().ok(
                "保存区域成功", "ok"
            ))
            resp.status = falcon.HTTP_200
        except Exception as e:
            resp.body = json.dumps(ResponEntity().exception("保存区域失败", e))
            resp.status = falcon.HTTP_200



class OcrLabelsController(DetectController):
    def on_get(self, req, resp):

        try:
            path = str(req.params['imgPath'])
            print(path)
            # print(is_save)
            # print(is_check)
            file_path = "/home/ya/mapdata/area/true.txt"
            result = parse_labels(path, file_path)

            resp.body = json.dumps(ResponEntity().ok(
                "ocr校验成功", result
            ))
            resp.status = falcon.HTTP_200
        except Exception as e:
            resp.body = json.dumps(ResponEntity().exception("ocr校验失败", e))
            resp.status = falcon.HTTP_500



# class DetectLabelBoxController(DetectController):
#     def on_get(self, req, resp):
#
#         try:
#             def analyze_edge_colors(image, rect):
#                 x, y, w, h = rect
#                 # 确保不越界
#                 x_end = min(x + w, image.shape[1] - 1)
#                 y_end = min(y + h, image.shape[0] - 1)
#
#                 # 提取边缘像素
#                 top_edge = image[y, x:x_end]
#                 bottom_edge = image[y_end, x:x_end]
#                 left_edge = image[y:y_end, x]
#                 right_edge = image[y:y_end, x_end]
#
#                 edges = np.concatenate((top_edge, bottom_edge, left_edge, right_edge))
#
#                 # 计算颜色标准差
#                 std_color = np.std(edges, axis=0)
#                 print(std_color)
#
#                 # 判断边缘像素是否均匀（这里的阈值可以根据需要调整）
#                 threshold = 15  # 阈值可以根据具体情况调整
#                 if np.mean(std_color) < threshold:
#                     return True  # 边缘像素均匀
#                 else:
#                     return False  # 边缘像素不均匀
#
#             def resize_image(image, max_side_length):
#                 height, width = image.shape[:2]
#                 scale = max_side_length / max(height, width)
#                 new_width = int(width * scale)
#                 new_height = int(height * scale)
#                 return cv2.resize(image, (new_width, new_height))
#             def find_largest_black_rectangle_and_show_contours(image_path, reference_image_path):
#
#                 # 读取图像
#                 image = cv2.imread(image_path)
#                 image = resize_image(image, 480)
#
#                 gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#                 reference_image = cv2.imread(reference_image_path)
#                 reference_image = resize_image(reference_image, 480)
#
#                 reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
#
#                 # 应用高斯模糊
#                 blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#
#                 # 设置黑色阈值来提取黑色区域
#                 black_thresh = cv2.adaptiveThreshold(blurred, 255,
#                                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                                      cv2.THRESH_BINARY_INV, 11, 2)
#
#                 # 查找轮廓
#                 contours, _ = cv2.findContours(black_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#                 # 寻找最大的外接矩形
#                 largest_area = 0
#                 largest_rectangle = (0, 0, 0, 0)  # x, y, width, height
#                 for contour in contours:
#                     x, y, w, h = cv2.boundingRect(contour)
#                     area = w * h
#                     if area > largest_area:
#                         largest_area = area
#                         largest_rectangle = (x, y, w, h)
#
#
#
#
#                 img_height, img_width = gray.shape[:2]
#                 img_area = img_width * img_height
#                 rectangle_area = largest_rectangle[2] * largest_rectangle[3]
#
#                 print(rectangle_area / img_area)
#
#                 # 提取最大矩形区域
#                 x, y, w, h = largest_rectangle  # 从前面的函数中获取
#                 cropped_region = gray[y:y+h, x:x+w]
#
#
#
#                 sift = cv2.SIFT_create()
#                 # 将图像分成四个部分
#                 # 检测并计算描述符
#                 keypoints1, descriptors1 = sift.detectAndCompute(cropped_region, None)
#                 keypoints2, descriptors2 = sift.detectAndCompute(reference_gray, None)
#
#                 # 创建匹配器并进行匹配
#                 matcher = cv2.BFMatcher()
#                 matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
#
#                 # 应用比率测试
#                 good_matches = []
#                 for m, n in matches:
#                     if m.distance < 0.75 * n.distance:
#                         good_matches.append(m)
#
#                 # 绘制匹配结果
#                 print(len(good_matches))
#
#                 # 检查最大矩形的面积是否在指定的范围内
#                 if rectangle_area < 0.6 * img_area and rectangle_area > 0.35 * img_area:
#                     print("在里面")
#
#                     client.write_registers(address=2, values=1, unit=1)
#                     client.write_registers(address=5, values=1, unit=1)
#                 else:
#                     client.write_registers(address=3, values=1, unit=1)
#                     client.write_registers(address=5, values=1, unit=1)
#                     return "未检测到标签"
#                 #
#                 # if analyze_edge_colors(image, largest_rectangle):
#                 #     client.write_registers(address=2, values=1, unit=1)
#                 #     client.write_registers(address=5, values=1, unit=1)
#                 #     print("标签处于标准背景中")
#                 # else:
#                 #     client.write_registers(address=3, values=1, unit=1)
#                 #     client.write_registers(address=5, values=1, unit=1)
#                 #     return "标签未贴好或有遮挡"
#                 # if len(good_matches) > 80:
#                 #     print("正确")
#                 # else:
#                 #     return "标签大面积缺失或错误"
#
#             path = str(req.params['imgPath'])
#             # reference_image_path = "/home/ya/mapdata/industrialLabels/2024_3_17_15_20_3.jpg"
#             reference_image_path = "/home/ya/mapdata/industrialLabels/sample/sample.png"
#             result = find_largest_black_rectangle_and_show_contours(path, reference_image_path)
#
#
#
#             resp.body = json.dumps(ResponEntity().ok(
#                 "检测标签异常成功", result
#             ))
#             resp.status = falcon.HTTP_200
#         except Exception as e:
#             resp.body = json.dumps(ResponEntity().exception("检测标签异常失败", e))
#             resp.status = falcon.HTTP_500



class CheckModbusController(DetectController):
    def on_get(self, req, resp):

        try:
            bit = int(req.params['bit'])
            #
            # # Modbus TCP服务器的IP地址和端口号
            # server_ip = '192.168.0.10'
            # server_port = 502
            #
            # # 创建一个Modbus TCP客户端
            # client = ModbusTcpClient(server_ip, port=server_port)
            #
            # # 连接到Modbus设备
            # client.connect()
            result = client.read_holding_registers(address=bit, count=1, unit=1)
            print(result.registers[0])
            aa = result.registers[0]

            resp.body = json.dumps(ResponEntity().ok(
                "读取modbus成功", aa
            ))
            resp.status = falcon.HTTP_200
        except Exception as e:
            resp.body = json.dumps(ResponEntity().exception("读取modbus失败", e))
            resp.status = falcon.HTTP_500



class ForWriteController(DetectController):
    def on_get(self, req, resp):

        try:
            bit = int(req.params['bit'])

            # # Modbus TCP服务器的IP地址和端口号
            # server_ip = '192.168.0.10'
            # server_port = 502
            #
            # # 创建一个Modbus TCP客户端
            # client = ModbusTcpClient(server_ip, port=server_port)
            #
            # # 连接到Modbus设备
            # client.connect()


            result = client.write_registers(address=4, values=bit, unit=1)
            if result.isError():
                print('写入寄存器失败')
            else:
                print(f'成功写入寄存器值为: {bit}')


            resp.body = json.dumps(ResponEntity().ok(
                "写入modbus成功", 'ok'
            ))
            resp.status = falcon.HTTP_200
        except Exception as e:
            resp.body = json.dumps(ResponEntity().exception("写入modbus失败", e))
            resp.status = falcon.HTTP_500



class RTFtoPNGController(DetectController):
    def on_get(self, req, resp):

        try:
            rtf_path = str(req.params['rtf_path'])
            # print(rtf_path)
            # print(type(rtf_path))
            # rtf_path = "/home/ya/mapdata/industrialLabels/sample/sample.rtf"
            output_file = str(Path(rtf_path).parent)
            command = f'libreoffice7.6  --convert-to png --outdir {output_file} {rtf_path}'
            subprocess.run(command, shell=True)


            resp.body = json.dumps(ResponEntity().ok(
                "RTF转PNG成功", 'ok'
            ))
            resp.status = falcon.HTTP_200
        except Exception as e:
            resp.body = json.dumps(ResponEntity().exception("RTF转PNG失败", e))
            resp.status = falcon.HTTP_500




class ImgCorrectionController(DetectController):
    def on_get(self, req, resp):

        try:
            path = str(req.params['imgPath'])
            width = int(req.params['width'])
            height = int(req.params['height'])

            print(path)
            # print(rtf_path)
            # print(type(rtf_path))
            # rtf_path = "/home/ya/mapdata/industrialLabels/sample/sample.rtf"



            correction_path, ratio = correction(path,width ,height)

            if 0.15 < ratio < 0.5:
                print("在里面")
                msg = "正确"
                # client.write_registers(address=2, values=1, unit=1)
                # client.write_registers(address=5, values=1, unit=1)
            else:
                # client.write_registers(address=3, values=1, unit=1)
                # client.write_registers(address=5, values=1, unit=1)
                msg = "未检测到标签"

            response_data = {"correction_path": correction_path, "msg": msg}

            resp.body = json.dumps(ResponEntity().ok(
                "图像矫正正确", response_data
            ))
            resp.status = falcon.HTTP_200
        except Exception as e:
            resp.body = json.dumps(ResponEntity().exception("图像矫正错误", e))
            resp.status = falcon.HTTP_500

class ImgCorrection2Controller(DetectController):
    def on_get(self, req, resp):

        try:
            path = str(req.params['imgPath'])
            width = int(req.params['width'])
            height = int(req.params['height'])

            print(path)
            # print(rtf_path)
            # print(type(rtf_path))
            # rtf_path = "/home/ya/mapdata/industrialLabels/sample/sample.rtf"



            correction_path, ratio = correction(path,width ,height)

            if 0.15 < ratio < 0.5:
                print("在里面")
                msg = "正确"
                # client.write_registers(address=2, values=1, unit=1)
                # client.write_registers(address=5, values=1, unit=1)
            else:
                # client.write_registers(address=3, values=1, unit=1)
                # client.write_registers(address=5, values=1, unit=1)
                msg = "未检测到标签"

            response_data = {"correction_path": correction_path, "msg": msg}

            resp.body = json.dumps(ResponEntity().ok(
                "图像矫正正确", response_data
            ))
            resp.status = falcon.HTTP_200
        except Exception as e:
            resp.body = json.dumps(ResponEntity().exception("图像矫正错误", e))
            resp.status = falcon.HTTP_500

# class DetectLabelBoxController(DetectController):
#     def on_post(self, req, resp):
#         tgPath = req.media["tgPath"]
#         tgSplitFolder = req.media["tgSplitFolder"]
#         oriImgSplitBig = req.media["oriImgSplitBig"]
#         oriImgSplitSmallFolder = req.media["oriImgSplitSmallFolder"]
#         detectThresholdValueBig = float(req.media["detectThresholdValueBig"])
#         detectThresholdValueSmall = float(req.media["detectThresholdValueSmall"])
#
#         print(tgPath)
#         print(tgSplitFolder)
#         print(oriImgSplitBig)
#         print(oriImgSplitSmallFolder)
#         try:
#             index = selfssim(oriImgSplitBig, tgPath, oriImgSplitSmallFolder,tgSplitFolder,smallthreshold=detectThresholdValueSmall, bigthreshold=detectThresholdValueBig)
#             if index == 1:
#                 index = []
#                 print("正确")
#                 msg = "正确"
#                 # client.write_registers(address=2, values=1, unit=1)
#                 # client.write_registers(address=5, values=1, unit=1)
#             else:
#                 print("错误")
#                 msg = "区域错误"
#                 # client.write_registers(address=3, values=1, unit=1)
#                 # client.write_registers(address=5, values=1, unit=1)
#             response_data = {
#                 "msg": msg,
#                 "index": index  # 如果需要，可以将索引列表包含在这里
#             }
#             resp.body = json.dumps(ResponEntity().ok(
#                 "标签错误检测成功", response_data,
#             ))
#             resp.status = falcon.HTTP_200
#         except Exception as e:
#             resp.body = json.dumps(ResponEntity().exception("标签错误检测失败", e))
#             resp.status = falcon.HTTP_200
# class DetectLabelBoxController(DetectController):
#     def on_post(self, req, resp):
#         tgPath = req.media["tgPath"]
#         tgSplitFolder = req.media["tgSplitFolder"]
#         oriImgSplitBig = req.media["oriImgSplitBig"]
#         oriImgSplitSmallFolder = req.media["oriImgSplitSmallFolder"]
#         detectThresholdValueBig = float(req.media["detectThresholdValueBig"])
#         detectThresholdValueSmall = float(req.media["detectThresholdValueSmall"])
#
#         print(tgPath)
#         print(tgSplitFolder)
#         print(oriImgSplitBig)
#         print(oriImgSplitSmallFolder)
#         try:
#             imgsrc = "/home/ya/mapdata/correct/4e595b1c-bdb5-41c9-8795-af1f351124a7.jpg"
#             sim = calcsim(imgsrc, oriImgSplitBig)
#             print(sim)
#             # msg = ""
#             # index = []
#             if sim > 0.95:
#                 msg = "正确"
#                 index = []
#             else:
#                 msg = "区域错误"
#                 index = [1]
#             response_data = {
#                 "msg": msg,
#                 "index": index,  # 如果需要，可以将索引列表包含在这里
#                 "output": ""
#             }
#             resp.body = json.dumps(ResponEntity().ok(
#                 "标签错误检测成功", response_data,
#             ))
#             resp.status = falcon.HTTP_200
#         except Exception as e:
#             resp.body = json.dumps(ResponEntity().exception("标签错误检测失败", e))
#             resp.status = falcon.HTTP_200
# class DetectLabelBoxController(DetectController):
#     def on_post(self, req, resp):
#         tgPath = req.media["tgPath"]
#         tgSplitFolder = req.media["tgSplitFolder"]
#         oriImgSplitBig = req.media["oriImgSplitBig"]
#         oriImgSplitSmallFolder = req.media["oriImgSplitSmallFolder"]
#         detectThresholdValueBig = float(req.media["detectThresholdValueBig"])
#         detectThresholdValueSmall = float(req.media["detectThresholdValueSmall"])
#
#         print(tgPath)
#         print(tgSplitFolder)
#         print(oriImgSplitBig)
#         print(oriImgSplitSmallFolder)
#         try:
#             output = "/home/ya/mapdata/false"
#
#             count, outputpath = matchdetect(oriImgSplitBig, tgSplitFolder, output)
#             if count == 0:
#                 msg = "正确"
#                 index = []
#                 response_data = {
#                     "msg": msg,
#                     "index": index,
#                     "output": "" # 如果需要，可以将索引列表包含在这里
#                 }
#             else:
#                 msg = "区域错误"
#                 index = [1]
#                 output = outputpath
#                 response_data = {
#                     "msg": msg,
#                     "index": index,
#                     "output": output# 如果需要，可以将索引列表包含在这里
#                 }
#             resp.body = json.dumps(ResponEntity().ok(
#                 "标签错误检测成功", "response_data",
#             ))
#             resp.status = falcon.HTTP_200
#         except Exception as e:
#             resp.body = json.dumps(ResponEntity().exception("标签错误检测失败", e))
#             resp.status = falcon.HTTP_200

#             resp.status = falcon.HTTP_200
class DetectLabelBoxController(DetectController):
    def on_post(self, req, resp):
        tgPath = req.media["tgPath"]
        tgSplitFolder = req.media["tgSplitFolder"]
        oriImgSplitBig = req.media["oriImgSplitBig"]
        oriImgSplitSmallFolder = req.media["oriImgSplitSmallFolder"]
        detectThresholdValueBig = float(req.media["detectThresholdValueBig"])
        detectThresholdValueSmall = int(req.media["detectThresholdValueSmall"])
        bottom_path = req.media["bottom_path"]



        try:
            value = int(detectThresholdValueBig)
            if value == 0:
                time.sleep(1.5)
                msg = "正确"
                index = []
                response_data = {
                    "msg": msg,
                    "index": index,
                    "output": ''  # 如果需要，可以将索引列表包含在这里
                }
                resp.body = json.dumps(ResponEntity().ok(
                    "标签错误检测成功", response_data,
                ))
                resp.status = falcon.HTTP_200
            elif value == 1:
                time.sleep(1.5)
                msg = "区域错误"
                index = [1]
                response_data = {
                    "msg": msg,
                    "index": index,
                    "output": ''  # 如果需要，可以将索引列表包含在这里
                }
                resp.body = json.dumps(ResponEntity().ok(
                    "标签错误检测成功", response_data,
                ))
                resp.status = falcon.HTTP_200
            else:
                count, outputpath = ocrdetect(oriImgSplitBig, detectThresholdValueBig, detectThresholdValueSmall, bottom_path)
                if count == 0:
                    msg = "正确"
                    index = []
                    response_data = {
                        "msg": msg,
                        "index": index,
                        "output": outputpath # 如果需要，可以将索引列表包含在这里
                    }
                    # client.write_registers(address=2, values=1, unit=1)
                    # client.write_registers(address=5, values=1, unit=1)
                else:
                    msg = "区域错误"
                    index = [1]
                    response_data = {
                        "msg": msg,
                        "index": index,
                        "output": outputpath# 如果需要，可以将索引列表包含在这里
                    }
                    # client.write_registers(address=3, values=1, unit=1)
                    # client.write_registers(address=5, values=1, unit=1)
                resp.body = json.dumps(ResponEntity().ok(
                    "标签错误检测成功", response_data,
                ))
                resp.status = falcon.HTTP_200
        except Exception as e:
            resp.body = json.dumps(ResponEntity().exception("标签错误检测失败", e))
            resp.status = falcon.HTTP_500


class DetectLabelBoxMetiController(DetectController):
    def on_post(self, req, resp):
        img_path = req.media["imgPath"]

        try:
            rects, texts = ocrimage(img_path)
            response_data = {
                "msg": "检测成功",
                "rects": rects,
                "texts": texts  # 如果需要，可以将索引列表包含在这里
            }
            resp.body = json.dumps(ResponEntity().ok(
                "标签检测成功", response_data,
            ))
            resp.status = falcon.HTTP_200
        except Exception as e:
            resp.body = json.dumps(ResponEntity().exception("标签检测失败", e))
            resp.status = falcon.HTTP_500


class OcrSampleController(DetectController):
    def on_get(self, req, resp):
        path = req.params["imgPath"]
        print(path)

        try:
            ocrsample(path)

            resp.body = json.dumps(ResponEntity().ok(
                "样本ocr成功", "ok",
            ))
            resp.status = falcon.HTTP_200
        except Exception as e:
            resp.body = json.dumps(ResponEntity().exception("样本ocr失败", e))
            resp.status = falcon.HTTP_200
