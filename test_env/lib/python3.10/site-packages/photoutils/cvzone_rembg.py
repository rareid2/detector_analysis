import cv2
import cvzone
import numpy as np
from PIL import Image as PILImage
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

blue_img = None

def blue_image(w, h):
	b = b'\x00\x00\xff' * w * h
	data = np.frombuffer(b, np.uint8)
	data = data.reshape(h, w, 3)
	im = PILImage.fromarray(data)
	# print('blue image, size=', im.size, w, h)
	return im

class CVZoneRembg:
	def __init__(self):
		self.segmentor = SelfiSegmentation()
		self.fpsReader = cvzone.FPS()

	def remove(self, img, bg_img=None):
		global blue_img
		w, h = img.size
		if bg_img is None:
			if not blue_img:
				blue_img = blue_image(w, h)
			bg_img = blue_img
		imgOut = self.segmentor.removeBG(img, bg_img, threshold=0.8)
		imgStack = cvzone.stackImages([img. imgOut], 2, 1)
		_, imgStack = self.fpsReader.update(imgStack)
		return imgStack
