"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import argparse
import cv2
import numpy as np
from src.utils import *
import models
from process import *
from PIL import Image
import time

# CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
# 		   'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
# 		   'tvmonitor']


def get_args():
	parser = argparse.ArgumentParser("You Only Look Once: Unified, Real-Time Object Detection")
	parser.add_argument("--image_size", type=int, default=448, help="The common width and height for all images")
	parser.add_argument("--conf_threshold", type=float, default=0.6)
	parser.add_argument("--nms_threshold", type=float, default=0.6)
	parser.add_argument("--pre_trained_model_type", type=str, choices=["model", "params"], default="model")
	parser.add_argument("--pre_trained_model_path_scene3", type=str, default="weights/yolo_model/whole_model_trained_yolo_bot")

	parser.add_argument('-a', '--arch_BOT', type=str, default='ResNet50_BOT_MultiTask', choices=models.get_names())
	# parser.add_argument('--resume_BOT', type=str, default='weights/bot_model/scene3/best_model.pth.tar', metavar='PATH')
	parser.add_argument('--resume_BOT', type=str, default='weights/bot_model/best_model.pth.tar', metavar='PATH')

	parser.add_argument("--input", type=str, default="./Test/video_in/1.avi",choices=[0,'./Test/video_in/1.avi'])
	parser.add_argument("--output", type=str, default="./Test/video_out/1.avi")

	parser.add_argument("--neg", type=str, default="./Test/neg",help="the folder where the  negative save")
	parser.add_argument("--ser_neg", type=str, default="./Test/ser_neg",help="the folder where the Seriously negative save")

	parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
	args = parser.parse_args()
	return args


def demo(opt):
	os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices
	model_yolo = Init_yolo(opt)
	model_BOT = Init_BOT_model(opt.resume_BOT, opt.arch_BOT)
	img_dir2 = "./2.jpg"
	# img = cv2.imread(img_dir2)

	cap = cv2.VideoCapture(opt.input)
	video_writer = cv2.VideoWriter(opt.output, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 15,(1280, 720))

	ret,img = cap.read()
	while ret:
		predictions ,width_ratio ,height_ratio,width,height = yolo2(opt,img,model_yolo)
		with_customer = False
		mark = 0  # 记录标志位，0代表不记录，1代表neg，2代表记录到ser_neg
		for pred in predictions[::-1]:
			print('=============================')
			xmin = int(max(pred[0] / width_ratio, 0))
			ymin = int(max(pred[1] / height_ratio, 0))
			xmax = int(min((pred[0] + pred[2]) / width_ratio, width))
			ymax = int(min((pred[1] + pred[3]) / height_ratio, height))
			box = (xmin,ymin,xmax,ymax)
			img2 = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
			img_crop = img2.crop(box)
			staff, customer, stand, sit, play_with_phone, male, female = BOT_recognition(model_BOT, img_crop)

			if staff:
				print('手机：', play_with_phone,'姿态：',sit,stand)
				if with_customer and (sit or play_with_phone):
					print("严重消极") # 严重消极黑色
					color = (0, 0, 0)
					name = "Seriously negative"
					mark = 2
				elif (not with_customer) and (sit or play_with_phone):
					print("一般消极")
					color = (144, 128, 112) #一般消极灰色
					name = "negative"
					if mark==0 :
						mark = 1
				else :
					print("积极行为")
					color = (0, 0, 255) # 积极行为红色
					name = "positive"

			else:
				print("客人")
				with_customer = True
				color = (204, 209, 72) # 客人蓝绿色
				name = "customer"

			img = Image_rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)),color, name)

		if mark == 1:
			nowtime = time.strftime('%m%d-%H%M', time.localtime(time.time()))
			file = os.path.join(opt.neg,str(nowtime)+".jpg")
			cv2.imwrite(file,img)
		elif mark == 2:
			nowtime = time.strftime('%m%d-%H%M', time.localtime(time.time()))
			file = os.path.join(opt.ser_neg,str(nowtime)+".jpg")
			cv2.imwrite(file,img)

		cv2.imshow('img',img)
		cv2.waitKey(5)
		video_writer.write(img)
		ret,img = cap.read()

	cap.release()
	cv2.destroyAllWindows()
	video_writer.release()

if __name__ == "__main__":
	opt = get_args()
	demo(opt)
