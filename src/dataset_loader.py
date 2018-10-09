from __future__ import print_function, absolute_import
import os
from PIL import Image
import numpy as np
import os.path as osp

import torch
from torch.utils.data import Dataset

def read_image(img_path):
	"""Keep reading image until succeed.
	This can avoid IOError incurred by heavy IO process."""
	got_img = False
	if not osp.exists(img_path):
		raise IOError("{} does not exist".format(img_path))
	while not got_img:
		try:
			img = Image.open(img_path).convert('RGB')
			got_img = True
		except IOError:
			print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
			pass
	return img


def read_bot_image(img_path,box):
	"""Keep reading image until succeed.
	This can avoid IOError incurred by heavy IO process."""
	got_img = False
	if not osp.exists(img_path):
		raise IOError("{} does not exist".format(img_path))
	while not got_img:
		try:
			img = Image.open(img_path).convert('RGB')
			img_crop = img.crop(box)
			got_img = True
		except IOError:
			print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
			pass
	return img_crop
class ImageDataset(Dataset):
	"""Image Person ReID Dataset"""
	def __init__(self, dataset, transform=None):
		self.dataset = dataset
		self.transform = transform

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		img_path, pid, camid = self.dataset[index]
		img = read_image(img_path)
		if self.transform is not None:
			img = self.transform(img)
		return img, pid, camid

class BOT_ImageDataset(Dataset):
	"""Image Person ReID Dataset"""
	def __init__(self, dataset, transform=None):
		self.dataset = dataset
		self.transform = transform

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		img_path, position0, position1, gender, staff, customer, stand, sit, phone = self.dataset[index]
		# print(img_path)
		box = (position0[0], position0[1], position1[0], position1[1])
		img = read_bot_image(img_path,box)
		# img.show()

		if self.transform is not None:
			img = self.transform(img)

		return img, gender, staff, customer, stand, sit,phone
class BOT_wuma_ImageDataset(Dataset):
	"""Image Person ReID Dataset"""
	def __init__(self, dataset, transform=None):
		self.dataset = dataset
		self.transform = transform

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		img_path, gender, staff, customer, stand, sit, phone = self.dataset[index]
		# print(img_path)
		# box = (position0[0], position0[1], position1[0], position1[1])
		# img = read_bot_image(img_path,box)
		img = read_image(img_path)
		# img.show()

		if self.transform is not None:
			img = self.transform(img)

		return img, gender, staff, customer, stand, sit, phone



class Load_person(Dataset):
	"""Load the cut person by yolo3"""
	def __init__(self, img, transform=None):
		self.img = img
		self.transform = transform

	def __len__(self):
		return 1
	def __getitem__(self, index):
		if self.transform is not None:
			img_trans = self.transform(self.img)
		return img_trans
