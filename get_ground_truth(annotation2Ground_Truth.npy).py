import os
import numpy as np
import re

'''
对于原始数据集 Test，如果是 /pos 中的，使用正则表达式读出每张图片的 Ground Truth 信息，
如果是 /neg 中的，直接给 空
对于每张图片的每个目标，其真实边界框由 4 个 int 表达，分别是 (Xmin, Ymin) - (Xmax, Ymax) 
PS：一行 存储一张图片中的 所有 边界框坐标，Test：741 行
以 .npy 格式保存在 
./Ground_Truth.npy
'''

def solve():
    Test_path = './data/Test/'
    annotations_path = './INRIAPerson/Test/annotations/'
    save_path = "./"
    save_file_name = 'Ground_Truth.npy'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    annotation_files = os.listdir(annotations_path)
    save_Ground_Truth = []
    for file in os.listdir(Test_path):
        if file.split('.')[0] == 'txt': 
            continue # 排除标签文件
        test_file_2_txt = file.split('.')[0] + '.txt'
        if test_file_2_txt in annotation_files:
            with open(annotations_path+test_file_2_txt, 'rb') as f:
                ann = f.read()
            Ground_Truth_list = re.findall('\(\d+, \d+\)[\s\-]+\(\d+, \d+\)', str(ann)) # like (250, 151) - (299, 294)
            coordinates = [re.findall('\d+', box) for box in Ground_Truth_list]
            coordinates = np.array(coordinates, dtype='int')
            save_Ground_Truth.append(coordinates)
        else :
            save_Ground_Truth.append(np.array([]))
    save_Ground_Truth = np.array(save_Ground_Truth)
    np.save(save_path+save_file_name, save_Ground_Truth)
    che = np.load(save_path + save_file_name, allow_pickle=True)
    for i in range(che.shape[0]):
        if (che[i] == save_Ground_Truth[i]).all():
            continue
        else :
            print(i)
    # print(che)

if __name__ == '__main__':
    solve()
