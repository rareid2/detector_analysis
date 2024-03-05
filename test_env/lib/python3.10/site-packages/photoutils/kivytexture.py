import numpy as np
def text2array(texture):
	image_data= np.array(bytearray(texture.pixels,
		dtype='uint8)
		.reshape(texture.height, texture.width, 4))
	return image_data

	
def array2texture(data):
	buf = data.tostring()
	h, w, c = data.shape
	if c == 3:
		c_fmt = 'rgb'
	else:
		c_fmt = 'rgba'
	texture = Texture.create(size=(w,h), colorfmt=c_fmt)
	texture.blit_buffer(buf, bufferfmt='ubyte', colorfmt=c_fmt)
	return texture

