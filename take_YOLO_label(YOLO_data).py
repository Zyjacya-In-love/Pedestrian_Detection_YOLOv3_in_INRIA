import os
import numpy as np
import re

'''
打标签 用于 darknet 训练
1. 对于原始数据集的 /pos，使用正则表达式读出每张图片的 Ground Truth 信息，
在 annotation 中，对于每张图片的每个目标，其真实边界框由 4 个 int 表达，分别是 (Xmin, Ymin) - (Xmax, Ymax) 
用于 训练 的标签 需要 用与图片同名的 txt 文件 存储一张图片中的 所有 边界框，每行一个，格式：
<object-class> <x> <y> <width> <height>
其中，x、y、width 和 height 是相对于图像的宽度和高度的，object-class 全为 0
2. 对于原始数据集的 /neg，与图片同名 txt 文件为空
PS：
/pos：Train：614 个，Test：288 个 需要标记
/neg：Train：1218 个，Test：453 个 不需要

yolov2中训练时数据要分为两部分：图片文件夹和labels文件夹。
但是在yolov3中每张图片的labels文件必须和图片放在同一目录下，否则训练的时候会提示找不到labels文件。

文件结构如下：
./data
    /Train -- 1832
    /Test -- 741
'''

def solve(mode):
    # /pos
    annotations_path = './INRIAPerson/{}/annotations/'.format(mode)
    save_path = "./data/{}/".format(mode)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    annotation_files = os.listdir(annotations_path)
    for file in annotation_files:
        with open(annotations_path+file, 'rb') as f:
            ann = f.read()
        img_size = re.findall('\d+ x \d+ x \d+', str(ann)) # like 594 x 720 x 3
        img_w, img_h = re.findall('\d+', img_size[0])[0:2]
        img_w = int(img_w)
        img_h = int(img_h)
        Ground_Truth_list = re.findall('\(\d+, \d+\)[\s\-]+\(\d+, \d+\)', str(ann)) # like (250, 151) - (299, 294)
        coordinates = [re.findall('\d+', box) for box in Ground_Truth_list]
        coordinates = np.array(coordinates, dtype='int')
        with open(save_path + file, 'w') as w:
            for pos in coordinates:
                x = (pos[0] + pos[2]) / 2.0
                y = (pos[1] + pos[3]) / 2.0
                width = pos[2] - pos[0]
                heigth = pos[3] - pos[1]
                newline = '0 '+ str(1.0*x/img_w)+' '+str(1.0*y/img_h) + ' '+str(1.0*width/img_w) + ' '+str(1.0*heigth/img_h)
                w.write(newline + '\n')
    # /neg
    neg_img_path = './INRIAPerson/{}/neg/'.format(mode)
    for file in os.listdir(neg_img_path):
        fp = open(save_path+file.split('.')[0] + '.txt', 'a')
        fp.close()

if __name__ == '__main__':
    solve('Train')
    solve('Test')
