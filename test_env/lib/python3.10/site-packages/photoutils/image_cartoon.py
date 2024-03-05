import time
from math import sqrt
from PIL import Image
from np_geometric import create_circle_mask, create_angles
import numpy as np
from np_image import NPImage

cartoon_classes = {
}

def CartoonRegister(name, klass):
	cartoon_classes[name] = klass

desc = {
	'cartoon_type':'Static',
	'options':{
	}
}

def cartoonBuild(desc):
	klass = cartoon_classes.get(desc['cartoon_type'])
	if klass is None:
		raise Exception('cartoon type not default')
	return klass(**desc.get('options', {}))

class Cartoon:
	def __init__(self, seconds=1, fps=120):
		self.seconds = seconds
		self.frame_cnt = fps * seconds
		self.cur_pos = 0

	def initial(self):
		self.cur_pos = 0

	def _get_frame(self):
		pass

	def get_frame(self):
		if self.cur_pos >= self.frame_cnt:
			raise StopIteration
		f = self._get_frame()
		self.cur_pos += 1
		return f
		
	def change_size(self, size):
		pass

	def set_fps(self, fps):
		frame_cnt = self.frame_cnt
		cur_pos = self.cur_pos
		self.frame_cnt = self.seconds * fps
		self.cur_pos = int(cur_pos * self.frame_cnt / frame_cnt)

	def __iter__(self):
		return self

	def __next__(self):
		x =  self.get_frame()
		return x

class ImageCartoon(Cartoon):
	def __init__(self, seconds=1, fps=120, imfile=None):
		super().__init__(seconds=seconds, fps=fps)
		self.npim = NPImage(size=(200,200), imfile=imfile)
							
	def change_size(self, size):
		self.npim.change_size(size)
		self.work_data = np.array(self.npim._resize(self.npim.image))
		self.sized_rate()

	def sized_rate(self):
		iw, ih = self.npim.image.size
		sw, sh = self.npim.size
		self.delta_x = self.delta_y = 0
		if sw / sh > iw / ih:
			self.rate = sh / ih
			zw = self.rate * iw
			self.delta_x = (sw - zw) / 2
		else:
			self.rate = sw / iw
			zh = self.rate * ih
			self.delta_y = (sh - zh) / 2

class StaticImage(ImageCartoon):
	def _get_frame(self):
		return self.npim.image

class FadeInImage(StaticImage):
	def _get_frame(self):
		# print(self.cur_pos, self.frame_cnt, self.npim.size)
		rate = self.cur_pos / self.frame_cnt
		d = self.work_data * rate
		return Image.fromarray(np.uint8(d))

class FadeOutImage(StaticImage):
	def _get_frame(self):
		# print(self.cur_pos, self.frame_cnt, self.npim.size)
		rate = 1 - self.cur_pos / self.frame_cnt
		d = self.work_data * rate
		return Image.fromarray(np.uint8(d))

class SwitchSmooth(ImageCartoon):
	def __init__(self, seconds=1, fps=120, 
							imfile=None, old_imfile=None):
		super().__init__(seconds=seconds, fps=fps, imfile=imfile)
		if old_imfile:
			self.old_npim = NPImage(size=(200,200), imfile=old_imfile)
		else:
			self.old_npim = None

	def change_size(self, size):
		super().change_size(size)
		if self.old_npim:
			self.old_npim.change_size(size)
			self.work_old_data = np.array(self.old_npim._resize(self.old_npim.image))
	def _get_frame(self): 
		rate = self.cur_pos / self.frame_cnt
		d = self.work_data * rate + self.work_old_data * (1 - rate)
		return Image.fromarray(np.uint8(d))

class Magnifier(ImageCartoon):
	def __init__(self, seconds=1, fps=120,
							imfile=None, box=[100, 100, 600, 800]):
		super().__init__(seconds=seconds, fps=fps, imfile=imfile)
		self.box = box
		w, h = self.npim.image.size
		self.left_rate = self.box[0] / self.frame_cnt
		self.top_rate =  self.box[1] / self.frame_cnt
		self.right_rate = (w - self.box[2]) / self.frame_cnt
		self.bottom_rate = (h - self.box[3]) / self.frame_cnt
	
	def crop_image(self, rate):
		w, h = self.npim.image.size
		left = int(self.box[0] - self.box[0] * rate)
		top = int(self.box[1] - self.box[1] * rate)
		right = int(self.box[2] + (w - self.box[2]) * rate)
		bottom = int(self.box[3] + (h - self.box[3]) * rate)
		crop_im = self.npim.image.crop((left, top, right, bottom))
		# return crop_im
		return self.npim._resize(crop_im)

class MagnifierOut(Magnifier):
	def _get_frame(self):
		rate = self.cur_pos / self.frame_cnt
		return self.crop_image(rate)
	
class MagnifierIn(Magnifier):
	def _get_frame(self):
		rate = 1 - self.cur_pos / self.frame_cnt
		return self.crop_image(rate)
	
class Circle(ImageCartoon):
	def __init__(self, seconds=1, fps=120,
							imfile=None, 
							radius=10, center=None, old_imfile=None):
		super().__init__(seconds=seconds, fps=fps, imfile=imfile)
		self._center = center
		w, h = self.npim.image.size
		if center is None:
			self._center = (int(w/2), int(h/2))
		self._radius = radius
		if old_imfile:
			self.old_npim = NPImage(size=(200,200), imfile=old_imfile)
		else:
			self.old_npim = None

	def change_size(self, size):
		super().change_size(size)
		if self.old_npim:
			self.old_npim.change_size(size)
			self.work_old_data = np.array(self.old_npim._resize(self.old_npim.image))
	def sized_rate(self):
		super().sized_rate()
		self.center = int(self._center[0] * self.rate + self.delta_x), \
						int(self._center[1] * self.rate + self.delta_y)
		self.radius = self.rate * self._radius
		self.max_radius = self.calc_max_radius()
		self.step = (self.max_radius - self.radius) / self.frame_cnt

	def calc_max_radius(self):
		w, h = self.npim.size
		corners = [ [self.delta_x,self.delta_y], 
					[self.delta_x, h - self.delta_y], 
					[w - self.delta_x, self.delta_y], 
					[w - self.delta_x, h - self.delta_y]]
		cx, cy = self.center
		return max([sqrt(abs(cx-x)**2 + abs(cy-y)**2) \
							for x, y in corners])

	def radius_circle(self, radius):
		# print(self.cur_pos, self.npim.size, self.center)
		w, h = self.npim.size
		tmp = np.zeros(w*h*3).reshape(h, w, 3)
		mask = create_circle_mask(h, w, radius, center=self.center)
		# mask = create_circle_mask(h, w, radius)
		# image = self.npim._resize(self.npim.image)
		# data = np.array(image)
		data = np.zeros((h, w, 3), np.uint8)
		if self.old_npim:
			# image1 = self.old_npim._resize(self.old_npim.image)
			# data1 = np.array(image1)
			data[:,:,0] = np.where(mask, self.work_data[:,:,0], self.work_old_data[:,:,0])
			data[:,:,1] = np.where(mask, self.work_data[:,:,1], self.work_old_data[:,:,1])
			data[:,:,2] = np.where(mask, self.work_data[:,:,2], self.work_old_data[:,:,2])
		else:
			data[:,:,0] = self.work_data[:,:,0] * mask
			data[:,:,1] = self.work_data[:,:,1] * mask
			data[:,:,2] = self.work_data[:,:,2] * mask

		return Image.fromarray(np.uint8(data))

class CircleIn(Circle):
	def _get_frame(self):
		radius = self.max_radius - self.step * self.cur_pos
		return self.radius_circle(radius)
	
class CircleOut(Circle):
	def _get_frame(self):
		radius = self.radius + self.step * self.cur_pos
		return self.radius_circle(radius)
		
class Arcs(ImageCartoon):
	def __init__(self, seconds=1, fps=120,
							imfile=None, 
							clockwise=True,
							other_point=(0,0),
							old_imfile=None, center=None):
		super().__init__(seconds=seconds, fps=fps, imfile=imfile)
		self.clockwise = clockwise
		self._other_point=other_point
		if old_imfile:
			self.old_npim = NPImage(size=(200,200), imfile=old_imfile)
		else:
			self.old_npim = None
		if center:
			self._center = center
		else:
			w, h = self.npim.image.size
			self._center = w // 2, h // 2

	def change_size(self, size):
		super().change_size(size)
		if self.old_npim:
			self.old_npim.change_size(size)
			self.work_old_data = np.array(self.old_npim._resize(self.old_npim.image))
		w, h = self.npim.size
		print(self.npim.size, self.center, self._center, self.npim.image.size, self.delta_y, self.delta_x)
		self.angles = create_angles(h, w, self.center, 
							other_point=self.other_point,
							clockwise=self.clockwise)
		self.minv = np.amin(self.angles)
		self.maxv = np.amax(self.angles)
		self.step = (self.maxv - self.minv) / self.frame_cnt
		print(self.__class__.__name__, 'change_size() called')

	def sized_rate(self):
		super().sized_rate()
		self.other_point = int(self._other_point[0] * self.rate + self.delta_x), \
						int(self._other_point[1] * self.rate + self.delta_y)
		self.center = int(self._center[0] * self.rate + self.delta_x), \
						int(self._center[1] * self.rate + self.delta_y)

	def _get_frame(self):
		angle =  self.cur_pos * self.step + self.minv
		mask = self.angles <= angle
		d = np.zeros(self.work_data.shape)
		if self.old_npim:
			d[:,:,0] = np.where(mask, self.work_data[:,:,0], self.work_old_data[:,:,0])
			d[:,:,1] = np.where(mask, self.work_data[:,:,1], self.work_old_data[:,:,1])
			d[:,:,2] = np.where(mask, self.work_data[:,:,2], self.work_old_data[:,:,2])
		else:
			d[:,:,0] = self.work_data[:,:,0] * mask
			d[:,:,1] = self.work_data[:,:,1] * mask
			d[:,:,2] = self.work_data[:,:,2] * mask
		return Image.fromarray(np.uint8(d))

class SlidingDoor(ImageCartoon):
	def __init__(self, seconds=1, fps=120,
							imfile=None, 
							old_imfile=None, center=None, direct="l-r"):
		super().__init__(seconds=seconds, fps=fps, imfile=imfile)
		if old_imfile:
			self.old_npim = NPImage(size=(200,200), imfile=old_imfile)
		else:
			self.old_npim = None
		self.dir = direct

	def change_size(self, size):
		super().change_size(size)
		if self.old_npim:
			self.old_npim.change_size(size)
			self.work_old_data = np.array(self.old_npim._resize(self.old_npim.image))
		w, h = self.npim.size
		if self.dir in ['l-r', 'r-l']:
			self.step = w / self.frame_cnt
		elif self.dir in ['t-b', 'b-t']:
			self.step = h / self.frame_cnt
		else:
			self.step = 4

	def _get_frame(self):
		d = np.zeros(self.work_data.shape, np.uint8)
		h, w, c = self.work_data.shape
		x = int(self.step * self.cur_pos)
		if self.dir == 't-b':
			d[:x, :, :] = self.work_data[h-x:, :, :]
			if self.old_npim:
				d[x:, :, :] = self.work_old_data[x:,:,:]

		elif self.dir == 'b-t':
			d[h-x:, :, :] = self.work_data[:x, :, :]
			if self.old_npim:
				d[:x, :, :] = self.work_old_data[:x,:,:]

		elif self.dir == 'l-r':
			d[:, :x, :] = self.work_data[:, w-x:, :]
			if self.old_npim:
				d[:, x:, :] = self.work_old_data[:,x:,:]
		elif self.dir == 'r-l':
			d[:, w-x:, :] = self.work_data[:, :x, :]
			if self.old_npim:
				d[:, :x, :] = self.work_old_data[:,:x,:]
		else:
			print(' --void-- ', self.dir)

		return Image.fromarray(np.uint8(d))

	
CartoonRegister('Static', StaticImage)
CartoonRegister('FadeIn', FadeInImage)
CartoonRegister('FadeOut', FadeOutImage)
CartoonRegister('SwitchSmooth', SwitchSmooth)
CartoonRegister('MagnifierOut', MagnifierOut)
CartoonRegister('MagnifierIn', MagnifierIn)
CartoonRegister('CircleIn', CircleIn)
CartoonRegister('CircleOut', CircleOut)
CartoonRegister('Arcs', Arcs)
CartoonRegister('SlidingDoor', SlidingDoor)
