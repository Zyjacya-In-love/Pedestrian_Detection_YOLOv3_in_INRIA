import os
import shutil

'''
制作 YOLO 训练 测试 需要的原始数据
将 ./INRIAPerson/Train 和 /Test 中 的 /pos /neg 都 拷贝 到一个文件夹 ./data/
文件结构如下：
./data
    /Train -- 1832
    /Test -- 741
'''

def solve(mode1, mode2):
    img_path = './INRIAPerson/{}/{}/'.format(mode1, mode2)
    save_path = "./data/{}/".format(mode1)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for file in os.listdir(img_path):
        src = img_path+file
        des = save_path+file
        shutil.copyfile(src, des)


if __name__ == '__main__':
    solve('Train', 'pos')
    solve('Train', 'neg')
    solve('Test', 'pos')
    solve('Test', 'neg')
