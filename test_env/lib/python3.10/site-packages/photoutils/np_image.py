import time
import numpy as np
from PIL import Image

class ImageSize:
	def __init__(self, size):
		w, h = size
		self.width = int(w)
		self.height = int(h)
		self.size = self.width, self.height

	def max_size(self, size):
		w, h = size
		sw, sh = self.size
		if (sw / sh) > (w / h):
			nh = sh
			nw = int(w / h * nh)
			return nw, nh
		else:
			nw = sw
			nh = int(h / w * nw)
			return nw, nh

class NPImage(ImageSize):
	def __init__(self, size=(200, 200), image=None, imfile=None):
		super().__init__(size)
		self.image = image
		self.imfile = imfile
		self.data = None
		if imfile:
			self.image = Image.open(imfile)
		# self.resize()
		self.data = np.array(self.image)

	def change_size(self, size):
		self.set_size(size)

	def set_size(self, size):
		self.size = (int(size[0]), int(size[1]))

	def resize(self):
		image = self._resize(self.image)
		self.data = np.array(image)

	def _resize(self, image):
		tim0 = time.time()
		nw, nh = self.max_size(image.size)
		tim1 = time.time()
		img = image.resize((nw, nh))
		# image.resize cost 0.0xxx seconds
		tim2 = time.time()
		img = self._fill(img)
		tim3 =  time.time()
		# print('resize=', tim2 -tim1, 'fill=', tim3 - tim2)
		return img

	def _fill(self, image):
		w, h = image.size
		d = None
		if w < self.size[0]:
			x = self.size[0] - w
			r1 = x // 2
			r2 = x - r1
			d = np.array(image)
			d1 = np.uint8(np.zeros(h*r1*3).reshape(h, r1, 3))
			d2 = np.uint8(np.zeros(h*r2*3).reshape(h, r2, 3))
			d = np.concatenate((d1,d,d2), axis=1)
		else:
			x = self.size[1] - h
			c1 = x // 2
			c2 = x - c1
			d = np.array(image)
			d1 = np.uint8(np.zeros(c1 * w * 3).reshape(c1, w, 3))
			d2 = np.uint8(np.zeros(c2 * w * 3).reshape(c2, w, 3))
			d = np.concatenate((d1,d,d2), axis=0)
		return Image.fromarray(d)

	def to_image(self):
		return Image.fromarray(self.data)

	def save(self, imfile=None):
		img = self.to_image()
		if imfile is None:
			imfile = self.imfile
		img.save(imfile)

	def gray_img(self):
		img = self._resize(self.image)
		return img.convert('L')

	def binary_img(self, thresh=128):
		maxval = 255
		img = self.gray_img()
		d = np.array(img)
		im_bin = (d >= thresh) * maxvall
		return Image.fromarray(np.uint8(im_bin))

	def color_binary_img(self, thresh=128):
		r , g, b = 255, 128, 32
		im_gray = np.array(self.gray_img())
		im_bool = im_gray >= thresh
		im_dst = np.empty((*im_gray.shape, 3))
		im_dst[:, :, 0] = im_bool * r
		im_dst[:, :, 1] = im_bool * g
		im_dst[:, :, 2] = im_bool * b
		return Image.fromarray(np.uint8(im_dst))

	def trim_img(self, x, y, width, height):
		data = np.array(self.image)
		trim_data = data[y:y+height, x:x+width]
		return self._resize(Image.fromarray(trim_data))

	def blend_img(self, other_im, rate):
		im = self._resize(self.image)
		nim = self._resize(other_im)
		data = np.array(im)
		odata = np.array(nim)
		d = data * rate + odata * (1 - rate)
		return Image.fromarray(d.astype(np.uint8))

	def mask_img(self, mask_im):
		im = self._resize(self.image)
		data = np.array(im)
		mim = self._resize(mask_im)
		mdata = np.array(mim)
		mdata = mdata // 168
		d = self.data * mdata
		return Image.fromarray(d.astype(np.uint8))

	def edge(self, thresh=128):
		b_n_w = np.array(self.binary_img(thresh=thresh))
		h,w = b_n_w.shape
		h_edge = self.hor_edge(w, h, b_n_w)
		v_edge = self.ver_edge(w, h, b_n_w)
		d = np.sqrt((v_edge **2) + (h_edge ** 2))
		return Image.fromarray(d.astype(np.uint8))

	def hor_edge(self, w,h,big_matrix):
		try:
			size_of_square_matrix = 3 #standard size of the submatrix
			vef = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
			#Calculating the total number of possible square submatrices can be formed from the given matrix and the output matrix.
			no_of_square_matrices = ((w-size_of_square_matrix)+1)*((h-size_of_square_matrix)+1)
			sosm=size_of_square_matrix
			row=(h-size_of_square_matrix)+1                     #It is the number of rows it will cover to find out the desired output.
			column=(w-size_of_square_matrix)+1                  #It is the number of columns it will cover to find out the desired output.
			result = []
			r = (h-3)+1                                         #row value of vertical edge filter
			c = (w-3)+1                                         #column value of vertical edge filter
			for i in range(row):
				for j in range(column):
					sq = big_matrix[i:i+sosm,j:j+sosm]
					sum = 0
					for k in range(3):
						for l in range(3):
							sum += (sq[k,l] * vef[k,l])
					result.append(sum)
			result_matrix = np.asarray(result).reshape(r,c)     #reshaping the resultant matrix
			return result_matrix

		except Exception as e:
			print("Invalid Input, Try again",str(e))

	def vert_edge(self, w,h,big_matrix):
		try:
			size_of_square_matrix = 3 #standard size of the submatrix
			vef = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
			#Calculating the total number of possible square submatrices can be formed from the given matrix and the output matrix.
			no_of_square_matrices = ((w-size_of_square_matrix)+1)*((h-size_of_square_matrix)+1)
			sosm=size_of_square_matrix
			row=(h-size_of_square_matrix)+1                     #It is the number of rows it will cover to find out the desired output.
			column=(w-size_of_square_matrix)+1                  #It is the number of columns it will cover to find out the desired output.
			max_val=0
			result = []
			r = (h-3)+1                                         #row value of horizontal edge filter
			c = (w-3)+1                                         #column value of horizontal edge filter

			for i in range(row):
				for j in range(column):
					sq = big_matrix[i:i+sosm,j:j+sosm]
					sum = 0
					for k in range(3):
						for l in range(3):
							sum += (sq[k,l] * vef[k,l])
					result.append(sum)
			result_matrix = np.asarray(result).reshape(r,c)     #reshaping the resultant matrix
			return result_matrix

		except Exception as e:
			print("Invalid Input, Try again",str(e))
