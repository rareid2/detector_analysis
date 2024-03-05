import time
import mediapipe as mp
import numpy as np
from PIL import Image as PILImage
from ffpyplayer.pic import Image as FFImage

mp_selfie_segmentation = mp.solutions.selfie_segmentation

class MediapipeRembg:
	def __init__(self):
		self.selfie_seg = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

	def blue_image(self, w, h):
		b = b'\x00\x00\xff' * w * h
		data = np.frombuffer(b, np.uint8)
		data = data.reshape(h, w, 3)
		return data
		
	def remove(self, img:np.ndarray, bg_img:np.ndarray=None):
		"""
		img and bg_img is type of numpy.ndarray
		"""
		if bg_img is None:
			h, w, c = img.shape
			bg_img = self.blue_image(w, h)
		results = self.selfie_seg.process(img)
		condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
		out_img = np.where(condition, img, bg_img)
		return out_img

	def remove_pil(self, img:PILImage, bg_img:PILImage=None):
		if bg_img:
			bg_img = np.asarray(bg_img)
		img = np.asarray(img)
		d = self.remove(img, bg_img=bg_img)
		return PILImage.fromarray(np.uint8(d))

	def remove_ff(self, img:FFImage, bg_img:FFImage=None):
		t1 = time.time()
		w, h = img.get_size()
		img = self.ffimage2npimage(img)
		if bg_img:
			bg_img = selfffimage2npimage(bg_img)
		d = self.remove(img, bg_img=bg_img)
		# d = img
		img = self.npimage2ffimage(d)
		t2 = time.time()
		# print('time cost=', t2 - t1, w, h, img.get_size())
		return img
		
	def ffimage2npimage(self, img:FFImage):
		w, h = img.get_size()
		ba = img.to_bytearray()[0]
		bb = bytes(ba)
		self.bbbba = bb
		img = np.frombuffer(bb, dtype=np.uint8)
		img = img.reshape((h, w, 3))
		return img

	def npimage2ffimage(self, img:np.ndarray):
		h, w, c = img.shape
		self.bbbbb = img.tobytes()
		d = bytearray(self.bbbbb)
		img = FFImage(plane_bufers=[d], pix_fmt='rgb24', size=(w, h))
		return img
