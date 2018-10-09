import models
import torch
import torch.nn as nn
import src.transforms as T
from src.dataset_loader import Load_person
from torch.utils.data import DataLoader
from src.utils import *
import cv2
import numpy as np

CLASSES = ['person']

'''
功能：加载yolo模型
'''
def Init_yolo(opt):
	print('load yolo weight...')
	if torch.cuda.is_available():
		model_scene3 = torch.load(opt.pre_trained_model_path_scene3)
	else:
		model_scene3 = torch.load(opt.pre_trained_model_path_scene3, map_location=lambda storage, loc: storage)
	model_scene3.eval()
	print('success load')

	return model_scene3

'''
功能：加载yolo模型
'''
def Init_bot(opt):
	print('load bot weight...')
	if torch.cuda.is_available():
		model_scene3 = torch.load(opt.pre_trained_model_path_scene3)
	else:
		model_scene3 = torch.load(opt.pre_trained_model_path_scene3, map_location=lambda storage, loc: storage)
	model_scene3.eval()
	print('success load')

	return model_scene3

'''
输入：
	1）BOT_dir:模型存放地址（pytorch）
	2）BOT_cfg:base模型类别，如resnet18,resnet50等
输出：
	模型
'''
def Init_BOT_model(BOT_dir,BOT_cfg):
	CUDA = torch.cuda.is_available()
	print("Initializing model: {}".format(BOT_cfg))
	model = models.init_model(name=BOT_cfg, num_classes=70, loss={'xent'}, use_gpu=CUDA)
	if BOT_dir:
		print("Loading checkpoint from '{}'".format(BOT_dir))
		checkpoint = torch.load(BOT_dir)
		model.load_state_dict(checkpoint['state_dict'])
		print("BOT Network successfully loaded")
	if CUDA:
		model = nn.DataParallel(model).cuda()
	return model

'''
把yolo的输出绘制在图片上
'''
def Yolo_rectangle(img,predictions,width_ratio,height_ratio,width,height):
	num = 0
	color = (0, 0, 255)
	for pred in predictions:
		num += 1
		xmin = int(max(pred[0] / width_ratio, 0))
		ymin = int(max(pred[1] / height_ratio, 0))
		xmax = int(min((pred[0] + pred[2]) / width_ratio, width))
		ymax = int(min((pred[1] + pred[3]) / height_ratio, height))

		if pred[5] == 'person' and not(xmax - xmin <50 and ymax - ymin <50):
			img = Image_rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)),color, num)

	return img


'''
给图片加框和名字
'''
def Image_rectangle(img,c1,c2,color,name):
	c1 = tuple(c1)
	c2 = tuple(c2)
	# color = (0, 0, 255)
	cv2.rectangle(img, c1, c2, color,1)    # 加框
	t_size = cv2.getTextSize(str(name), cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
	c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
	cv2.rectangle(img, c1, c2, color, -1) # -1填充作为文字框底色
	cv2.putText(img, str(name), (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
	cv2.putText(img, str(name), (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
	return img


'''
对图片进行行人检测
'''
def yolo2(opt,image,model):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	height, width = image.shape[:2]
	image = cv2.resize(image, (opt.image_size, opt.image_size))
	image = np.transpose(np.array(image, dtype=np.float32), (2, 0, 1))
	image = image[None, :, :, :]
	width_ratio = float(opt.image_size) / width
	height_ratio = float(opt.image_size) / height
	data = Variable(torch.FloatTensor(image))
	if torch.cuda.is_available():
		data = data.cuda()
	with torch.no_grad():
		logits = model(data)
		predictions = post_processing(logits, opt.image_size, CLASSES, model.anchors, opt.conf_threshold,
		                              opt.nms_threshold)
	if len(predictions) != 0:
		predictions = predictions[0]

	return predictions ,width_ratio ,height_ratio,width,height

'''
输入：
	1）model_BOT:BOT多任务识别模型（pytorch）
	2）img:人脸裁剪图片
输出：

'''
def BOT_recognition(model_BOT,img):

	transform = T.Compose([
		T.Resize((256, 128)),
		T.ToTensor(),
		T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	loader = DataLoader(
		Load_person(img, transform=transform),
		batch_size=1, shuffle=False, num_workers=0,
		pin_memory=True, drop_last=False, )
	model_BOT.eval()

	with torch.no_grad():
		for batch_idx,img2 in enumerate(loader):
			if torch.cuda.is_available():img2 = img2.cuda()
			# score = model_BOT(img2)
			gender_outputs, staff_outputs, customer_outputs, stand_outputs, sit_outputs, phone_outputs = model_BOT(img2)

			staff_con = float(Confidence(staff_outputs, 1))
			customer_con = float(Confidence(customer_outputs, 1))
			stand_con =float(Confidence(stand_outputs, 1))
			sit_con = float(Confidence(sit_outputs, 1))
			play_with_phone_con = float(Confidence(phone_outputs, 1))
			male_con = float(Confidence(gender_outputs, 0))
			female_con = float(Confidence(gender_outputs, 1))

			staff_con = 1 if  staff_con>0.5 else 0
			customer_con = 1 if  customer_con>0.5 else 0
			stand_con = 1 if  stand_con>0.5 else 0
			sit_con = 1 if  sit_con>0.5 else 0
			play_with_phone_con = 1 if  play_with_phone_con>0.1 else 0
			male_con = 1 if  male_con>0.5 else 0
			female_con = 1 if  female_con>0.5 else 0


	return staff_con,customer_con,stand_con,sit_con,play_with_phone_con,male_con,female_con

def Confidence(fc_outputs, assign):
	a = fc_outputs[0][0].cpu().numpy()
	b = fc_outputs[0][1].cpu().numpy()

	if assign == 0:
		con = np.e ** a / (np.e ** a + np.e ** b)
	else:
		con = np.e ** b / (np.e ** a + np.e ** b)

	return '%.6f' % con
