# -*- coding: utf-8 -*-
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

class ContrastStretch:
    def __init__(self):
        self.img = ""
        self.output_img = ""
        self.img_r = ""  # r通道
        self.img_g = ""  # g通道
        self.img_b = ""  # b通道
        self.img_h_r = ""  # 均衡化后的r通道
        self.img_h_g = ""
        self.img_h_b = ""
        self.img_path = './data/images/'
        self.img_enhance_path = './data/images_enhance/'

    def stretch(self, input_image):
        self.img = cv2.imread(input_image)
        self.img_b = self.img[:, :, 0]
        self.img_g = self.img[:, :, 1]
        self.img_r = self.img[:, :, 2]

        self.img_h_b = cv2.equalizeHist(self.img_b)
        self.img_h_g = cv2.equalizeHist(self.img_g)
        self.img_h_r = cv2.equalizeHist(self.img_r)
        self.output_img = cv2.merge((self.img_h_b, self.img_h_g, self.img_h_r))

        return self.output_img


    def PlotHistogram(self):
        ax = plt.subplot(231)
        ax.set_title("img_b")
        x1, y1, z1 = plt.hist(self.img_b.ravel(), bins=256, range=(0, 256), label="img_b")
        ax = plt.subplot(232)
        ax.set_title("img_g")
        x2, y2, z2 = plt.hist(self.img_g.ravel(), bins=256, range=(0, 256), label='img_g')
        ax = plt.subplot(233)
        ax.set_title("img_r")
        x3, y3, z3 = plt.hist(self.img_r.ravel(), bins=256, range=(0, 256), label='img_r')
        plt.show()

    def ShowImages(self):
        '''
        显示图片
        :return:
        '''
        plt.imshow(self.output_img[:, :, ::-1])
        plt.show()

    def Multi_Process(self, test_path):
        '''
        对所有图像进行直方图均衡化操作并进行保存
        :param test_path: 图像增强的保存文件夹
        :return: 直接保存图片于test_path处
        '''
        # 创建新文件夹存储图片
        if not os.path.exists(test_path):
            os.mkdir(test_path)
        file_list = []  # 直接用生成的模板匹配
        # 提取n（用_作为分隔）
        # n = []
        # for file in file_list:
        #     num_file_list = []
        #     num = re.sub("\\D", " ", file).split()[0]
        #     if int(num) == file[0:2] or int(num) == file[0:3]:
        #         num_file_list.append(file)
        #
        # 思路：直接生成列表，之后查询是否有此图。
        for i in range(0, 60):
            for j in range(0, 600):
                file_list.append(str(i+1)+'_'+str(j+1)+'.jpg')
        for index, file in enumerate(tqdm(file_list)):
            if os.path.exists('./data/images/' + file):
                output_img = ContrastStretch.stretch(self, './data/images/' + file)
                cv2.imwrite(test_path + file, output_img)
            else:
                print("缺失图片：", file)


if __name__ == '__main__':
    stretcher = ContrastStretch()
    stretcher.Multi_Process('./data/images_enhance/')
    # stretcher.stretch('./data/images/1_1.jpg')
