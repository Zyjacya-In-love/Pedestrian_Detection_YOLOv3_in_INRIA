import os
from collections import namedtuple
import numpy as np
import cv2
from skimage.io import imread, imsave

'''
评价 行人检测器 detector 的 检测质量（detection quality）
	主要是体现在输入图像中的行人能否被成功检测出来，以及得到的位置是否准确
使用 IoU(Intersection-over-Union) 这个指标来评价 检测器 的准确率，
若 Ground Truth 与 predict bb 的 IOU score > 0.5 即认为这个窗口是正确的
PS： 
将 Ground Truth 与 predict bb 一同画在原始图像上并存储在 ./vs 中
./vs
	/YOLO -- 741
'''
IoUThresh = 0.5

# define the `Detection` object
Detection = namedtuple("Detection", ["image_name", "gt", "pred"])

def bb_intersection_over_union(boxA, boxB):
	if boxA.shape[0] == 0 or boxB.shape[0] == 0:
		return 0
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou

def solve(mode):
	original_image_path = './data/Test/'
	save_path = "./vs/"+mode+'/'
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	original_image_files = os.listdir(original_image_path)

	Ground_Truth = np.load('./Ground_Truth.npy', allow_pickle=True)

	predict_path = "./predict/"
	predict_file_name = mode + '.npy'
	predict_bb = np.load(predict_path + predict_file_name, allow_pickle=True)
	data = []
	for i in range(len(original_image_files)):
                file = original_image_files[i]
                if file.split('.')[0] == 'txt': 
                    continue # 排除标签文件
		# img = original_image_path + original_image_files[i]
		data.append(Detection(original_image_files[i], Ground_Truth[i], predict_bb[i]))

	pred_num = 0
	gt_num = 0
	good = 0
	# loop over the example detections
	for detection in data:
		pred_num += np.shape(detection.pred)[0]
		gt_num += np.shape(detection.gt)[0]
		for pred in detection.pred:
			for gt in detection.gt:
				iou = bb_intersection_over_union(gt, pred)
				if iou > IoUThresh:
					good += 1
					break
		# load the image
		image = imread(original_image_path + detection.image_name)
		# draw the ground-truth bounding box along with the predicted bounding box
		for gt in detection.gt:
			cv2.rectangle(image, tuple(gt[:2]), tuple(gt[2:]), (0, 255, 0), 2) # green
		for pred in detection.pred:
			cv2.rectangle(image, tuple(pred[:2]), tuple(pred[2:]), (255, 0, 0), 2) # red
		# save the output image
		imsave(save_path+detection.image_name, image)

	print('Ground Truth sum number is', gt_num)
	print('predictions bb number is', pred_num)
	print('number of correct prediction is', good)
	print('Precision is', float(good * 1.0 / pred_num)) # 查准率（precision）=TP/(TP+FP)
	print('Recall is', float(good * 1.0 / gt_num)) # 查全率（Recall）=TP/(TP+FN)

if __name__ == '__main__':
	solve('YOLO')





