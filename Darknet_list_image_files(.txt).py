import os
import numpy as np
import re

'''
Darknet 需要 一个 txt 文件列出所有 需要训练的图片的位置，以及需要测试的图片的位置
文件结构如下：
./data
    train.txt
    test.txt
'''

def solve(mode):
    img_path = os.getcwd()+"/data/{}/".format(mode)
    save_path = "./data/"
    save_file = '{}.txt'.format(mode)
    with open(save_path + save_file, 'w') as w:
        for file in os.listdir(img_path):
            if file == 'labels':
                continue
            w.write(img_path + file + '\n')


if __name__ == '__main__':
    solve('Train')
    solve('Test')
