# 对预处理数据集的程序进行函数化
# 汇集所有的处理方式
import os
import cv2
import matplotlib.pyplot as plt
import re
import shutil
'''
文件夹处理部分：
file_folder_name_change()：对文件夹名字进行处理
file_name_change(file_folder, n_path):对文件名进行处理
file_move_to_one(file_folder):将所有的文件移动到一个文件夹
'''


def file_folder_name_change(p):
    # 特点：需要特定字符分开的情况
    # 用来处理文件夹的名字，使其变为数字累加
    file_list1 = os.listdir(p)
    file_list1_num = []
    for file in file_list1:
        file_list1_num.append(re.sub('\\D', '', file))
        os.rename(p + file, p + re.sub('\\D', '', file))


def file_name_change(file_folder, n_path):
    # 用来处理，遍历文件名
    path = './data/' + file_folder + n_path + '/'
    file_list = os.listdir(path)
    # 直接将其命名为类似于"1_1"：
    for i in range(len(file_list)):
        if file_list[i][-3:] == 'jpg':
            os.rename(path + file_list[i], path + n_path + "_" + str((i // 2) + 1) + '.jpg')
        elif file_list[i][-3:] == 'txt':
            os.rename(path + file_list[i], path + n_path + "_" + str((i // 2) + 1) + '.txt')
        else:
            print("Wrong!")


def file_move_to_one(file_folder, dst_path):
    path = file_folder
    file_list = os.listdir(path)
    # dst_path = './data/bbox2_all/'
    for f in file_list:
        folder_path = os.listdir(path + f)
        for k in folder_path:
            shutil.move(path+f+'/'+k, dst_path)
    print("done!")

'''
对于错误的txt进行处理
'''


def txt_detection(bbox_path, n_path):
    path = bbox_path + n_path
    n = re.sub("\\D", "", path)
    # 遍历txt
    txt_list = os.listdir(path)
    if int(n) < 10:
        txt_list = sorted(txt_list, key=lambda x: int(x[2:-4]))
    else:
        txt_list = sorted(txt_list, key=lambda x: int(x[3:-4]))
    for txt in txt_list:
        bbox = []
        with open(path + txt) as f:
            bbox.append(f.read().split())
        data_bbox = []
        with open('./data/data_set/' + n_path + txt) as f:
            data_bbox.append(f.read().split())
        if bbox[0]:  # 存在bbox时
            # 检测数据是否有误
            if bbox[0][0] != str(n) or len(bbox[0]) % 5 != 0:  # 第一列不是类别或长度出现问题的错误
                bbox_separator = [i for (i, v) in enumerate(bbox[0]) if v == n]
                data_bbox_separator = [i for (i, v) in enumerate(data_bbox[0]) if v == n]
                # 新建一个列表，用来存储匹配到的源数据
                rewrite_bbox = []
                for i in bbox_separator:
                    for j in range(len(data_bbox_separator)):
                        if bbox[0][i + 1] == data_bbox[0][j * 5 + 1]:
                            rewrite_bbox.append(data_bbox[0][5 * j:5 * j + 5])
                # rewrite_bbox = [k for l in rewrite_bbox for k in l]
                # 直接用源数据的特定行进行输出
                with open('./data/bbox/' + n_path + txt, 'w') as f:
                    for line in rewrite_bbox:
                        f.writelines(" ".join(line))
                        f.writelines("\n")


# 可以使用正确的txt刷新图片的包围盒
def refresh_img(data_path, n_path):
    start = 0
    file_list = os.listdir(data_path + n_path)
    n = re.sub("\\D", "", n_path)
    txt_list = []
    img_list = []
    for f in file_list:
        if f[-3:] == 'txt':
            txt_list.append(f)
        if f[-3:] == 'jpg':
            img_list.append(f)
    if int(n) < 10:
        txt_list = sorted(txt_list, key=lambda x: int(x[2:-4]))
        img_list = sorted(img_list, key=lambda x: int(x[2:-4]))
    else:
        txt_list = sorted(txt_list, key=lambda x: int(x[3:-4]))
        img_list = sorted(img_list, key=lambda x: int(x[3:-4]))
    for i in range(start, len(img_list)):
        # 读取txt:
        bbox = []
        with open('./data/bbox/' + n_path + txt_list[i]) as f:
            bbox.append(f.read().split())
        # 读图
        img = cv2.imread('./data/data_set/' + n_path + img_list[i])
        h = img.shape[0]
        w = img.shape[1]
        loop = int(len(bbox[0]) / 5)  # 需要循环几次
        j = 0
        while j < loop:
            x_center = float(bbox[0][1 + 5 * j]) * w
            y_center = float(bbox[0][2 + 5 * j]) * h
            width = float(bbox[0][3 + 5 * j]) * w
            height = float(bbox[0][4 + 5 * j]) * h
            # 求最左上角的坐标
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x1 + width)
            y2 = int(y1 + height)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
            plt.annotate(str(j), xy=(x1, y1))
            # plt.annotate(str(j)+str(bbox[0][1:]), xy=(x1, y1))
            j += 1
        cv2.imwrite(test_path + n_path + img_list[i], img)


'''
清洗数据：
process1():进行手动处理，剔除掉错误的bbox
process2():进行自动处理，去除最后一个bbox
'''


def img_process1(data_path, n_path, test_path):
    start = 0
    # 起始图片
    # 创建文件夹（不存在时）
    n = re.sub("\\D", "", n_path)
    if not os.path.exists(test_path + n_path):
        os.mkdir(test_path + n_path)
    if not os.path.exists(data_path + n_path):
        os.mkdir(data_path + n_path)
    # 思路：读入图片和标签，之后进行判断，判断完成以后输入到新的文件夹里。
    # 1、单独使用时使用此段
    file_list = os.listdir(data_path + n_path)
    txt_list = []
    img_list = []
    for f in file_list:
        if f[-3:] == 'txt':
            txt_list.append(f)
        if f[-3:] == 'jpg':
            img_list.append(f)
    if int(n) < 10:
        txt_list = sorted(txt_list, key=lambda x: int(x[2:-4]))
        img_list = sorted(img_list, key=lambda x: int(x[2:-4]))
    else:
        txt_list = sorted(txt_list, key=lambda x: int(x[3:-4]))
        img_list = sorted(img_list, key=lambda x: int(x[3:-4]))
    # 读入jpg和txt
    for i in range(start, len(img_list)):
        # 读取txt:
        bbox = []
        with open('./data/data_set/' + n_path + txt_list[i]) as f:
            bbox.append(f.read().split())
        # 读图
        img = cv2.imread('./data/data_set/' + n_path + img_list[i])
        h = img.shape[0]
        w = img.shape[1]
        loop = int(len(bbox[0]) / 5)  # 需要循环几次
        plt.figure(i)
        j = 0
        while j < loop:
            x_center = float(bbox[0][1 + 5 * j]) * w
            y_center = float(bbox[0][2 + 5 * j]) * h
            width = float(bbox[0][3 + 5 * j]) * w
            height = float(bbox[0][4 + 5 * j]) * h
            # 求最左上角的坐标
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x1 + width)
            y2 = int(y1 + height)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
            plt.annotate(str(j), xy=(x1, y1))
            # plt.annotate(str(j)+str(bbox[0][1:]), xy=(x1, y1))
            j += 1
        plt.imshow(img[:, :, ::-1])
        plt.show()
        # 删除
        # 支持多选，最多选8个
        num = input("第" + str(i + 1) + "张" + str(img_list[i]) + " " + "是否需要删除：(0/1/2/3/4/5/6/7/8/enter):")
        num = sorted(num, key=int, reverse=True)
        for n in num:
            if n == '0':
                bbox[0][0:5] = []
            if n == '1':
                bbox[0][5:10] = []
            if n == '2':
                bbox[0][10:15] = []
            if n == '3':
                bbox[0][15:20] = []
            if n == '4':
                bbox[0][20:25] = []
            if n == '5':
                bbox[0][25:30] = []
            if n == '6':
                bbox[0][30:35] = []
            if n == '7':
                bbox[0][35:40] = []
            if n == '8':
                bbox[0][40:45] = []

        # 将干净的数据直接标注存好：
        img = cv2.imread(data_path + n_path + img_list[i])
        loop = int(len(bbox[0]) / 5)  # 需要循环几次
        for j in range(int(loop)):
            x_center = float(bbox[0][1 + 5 * j]) * w
            y_center = float(bbox[0][2 + 5 * j]) * h
            width = float(bbox[0][3 + 5 * j]) * w
            height = float(bbox[0][4 + 5 * j]) * h
            # 求最左上角的坐标
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x1 + width)
            y2 = int(y1 + height)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
        # 保存图片
        cv2.imwrite(test_path + n_path + img_list[i], img)
        # 保存txt数据
        with open('./data/bbox/' + n_path + txt_list[i], 'w') as f:
            if len(bbox[0]) >= 10:  # 多个目标存在，则需要输入多行
                for l in range(loop):
                    f.writelines(" ".join(bbox[0][5 * l:5 + 5 * l]))
                    f.writelines('\n')
            else:
                f.write(" ".join(bbox[0]))
    print("done!")


def img_process2(data_path, n_path, test_path):
    path = data_path + n_path
    # 批量读入txt
    file_list = os.listdir(path)
    txt_list = []
    img_list = []
    for f in file_list:
        if f[-3:] == 'txt':
            txt_list.append(f)
        if f[-3:] == 'jpg':
            img_list.append(f)
    txt_list = sorted(txt_list, key=lambda x: int(x[2:-4]))
    img_list = sorted(img_list, key=lambda x: int(x[2:-4]))
    bbox_list = []
    start = 0
    end = 181
    # end = len(txt_list)
    for txt in txt_list:
        with open(path + txt, "r") as f:
            bbox_list.append(f.read().split())  # 通过空格进行分割
    # 读入图片并进行标注
    # 只保留0号bbox
    for i in range(start, end):
        img = cv2.imread(path + img_list[i])
        bbox = bbox_list[i]
        if len(bbox_list[i]) >= 10:
            bbox_list[i] = bbox_list[i][0:5]
        if bbox:
            x_center = float(bbox[1]) * 1280
            y_center = float(bbox[2]) * 720
            width = float(bbox[3]) * 1280
            height = float(bbox[4]) * 720
            # 求最左上角的坐标
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x1 + width)
            y2 = int(y1 + height)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 5)
            plt.annotate(str(0) + str(bbox[1:5]), xy=(x1, y1))
            # cv2.imwrite('./data/test_set/1/' + img_list[i], img)
            cv2.imwrite(test_path + n_path + img_list[i], img)
            print(str(i) + " done! ")
        else:
            print(str(i) + " 错误！请手动剔除数据")
        with open('./data/bbox/3/' + txt_list[i], 'w') as f:
            f.write(" ".join(bbox_list[i]) + '\n')  # 去掉bbox_list的括号、引号


if __name__ == '__main__':
    # path = './data/data_set/'
    # file_folder_name_change(path)
    # file_name_change('data_set/', '4')
    data_path = './data/data_set/'
    bbox_path = './data/bbox/'
    bbox2_path = './data/bbox2/'
    test_path = './data/test_set/'
    n_path = '1/'
    # txt_detection(bbox_path, n_path)
    # refresh_img(data_path, n_path)
    img_process1(data_path, n_path, test_path)
    # file_move_to_one(data_path, dst_path='./data/images/')
    print(n_path + " is done!")
