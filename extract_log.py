import inspect
import os
import random
import sys

'''
绘制 log 曲线，但是loss中存在 nan，去掉 nan
# 该文件用于提取训练log，去除不可解析的log后使log文件格式化，生成新的log文件供可视化工具绘图
from https://blog.csdn.net/qq_34806812/article/details/81459982
'''

def extract_log(log_file, new_log_file, key_word):
    with open(log_file, 'r') as f:
        with open(new_log_file, 'w') as train_log:
            for line in f:
                # 去除多GPU的同步log；去除除零错误的log
                if ('Syncing' in line) or ('nan' in line):
                    continue
                if key_word in line:
                    train_log.write(line)
    f.close()
    train_log.close()


extract_log('./training.log', './new_log_loss.txt', 'images')
# extract_log('./2048/train_log2.txt', 'log_iou2.txt', 'IOU')