import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
绘制 loss 曲线
from: https://blog.csdn.net/qq_34806812/article/details/81459982
'''

# 根据自己的log_loss.txt中的行数修改lines, 修改训练时的迭代起始次数(start_ite)和结束次数(end_ite)。
lines = 135249
start_ite = 1  # log_loss.txt里面的最小迭代次数
end_ite = 130000  # log_loss.txt里面的最大迭代次数
step = 10  # 跳行数，决定画图的稠密程度
igore = 0  # 当开始的loss较大时，你需要忽略前igore次迭代，注意这里是迭代次数

# y_ticks = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]  # 纵坐标的值，可以自己设置。
data_path = './new_log_loss.txt'  # log_loss的路径。
result_path = './avg_loss'  # 保存结果的路径。

####-----------------只需要改上面的，下面的可以不改动
names = ['loss', 'avg', 'rate', 'seconds', 'images']
result = pd.read_csv(data_path, skiprows=[x for x in range(lines) if
                                          (x < lines * 1.0 / ((end_ite - start_ite) * 1.0) * igore or x % step != 9)],
                     error_bad_lines= \
                         False, names=names)
result.head()
for name in names:
    result[name] = result[name].str.split(' ').str.get(1)

result.head()
result.tail()

for name in names:
    result[name] = pd.to_numeric(result[name])
result.dtypes
print(result['avg'].values)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

###-----------设置横坐标的值。
x_num = len(result['avg'].values)
tmp = (end_ite - start_ite - igore) / (x_num * 1.0)
x = []
for i in range(x_num):
    x.append(i * tmp + start_ite + igore)
# print(x)
print('total = %d\n' % x_num)
print('start = %d, end = %d\n' % (x[0], x[-1]))
###----------


ax.plot(x, result['avg'].values, label='avg_loss')
# ax.plot(result['loss'].values, label='loss')
# plt.yticks(y_ticks)  # 如果不想自己设置纵坐标，可以注释掉。
plt.grid()
ax.legend(loc='best')
ax.set_title('The loss curves')
ax.set_xlabel('batches')
fig.savefig(result_path)
# fig.savefig('loss')