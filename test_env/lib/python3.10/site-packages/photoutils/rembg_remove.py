
from rembg import remove, new_session
from PIL import Image as PILImage
import numpy as np

rm_session = None
blue_img = None

def blue_image(w, h):
	b = b'\x00\x00\xff' * w * h
	data = np.frombuffer(b, np.uint8)
	data = data.reshape(h, w, 3)
	im = PILImage.fromarray(data)
	# print('blue image, size=', im.size, w, h)
	return im

def do_remove(img, bg_img=None):
	# print('here1' , time.time(), bg_img)
	t1 = time.time()
	global rm_session
	if rm_session is None:
		rm_session = new_session()
	# print('here2', time.time())
	w, h = img.size
	if bg_img is None:
		# print('blue background....')
		bg_img = blue_image(w, h)
	# print('here3', time.time())
	w1, h1 = bg_img.size
	if w != w1 or h != h1:
		# print('here3-1', time.time(), w, h, w1, h1)
		ibg_img = bg_img.resize((w, h))
		# print('here3-2', time.time())
	# print('here4', time.time())
	mask = remove(img, only_mask=True, session=rm_session)
	# print('mask type=', type(mask))
	# print('here5', time.time())
	d1 = np.array(img)
	d2 = np.array(bg_img)
	# print('here6', time.time())
	x = np.zeros((h, w, 3))
	x[:,:,0] = np.where(mask, d1[:,:,0], d2[:,:,0])
	x[:,:,1] = np.where(mask, d1[:,:,1], d2[:,:,1])
	x[:,:,2] = np.where(mask, d1[:,:,2], d2[:,:,2])
	# print('here7', time.time())
	ret = PILImage.fromarray(np.uint8(x))
	t2 = time.time()
	print('time cost=', t2 - t1)
	return ret

