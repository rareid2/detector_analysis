import numpy as np

def create_angles(h, w, center, other_point, clockwise=False):
	"""
	center = h, w
	"""
	a180 = 180
	if clockwise:
		a180 = -180
	y, x = other_point
	cy, cx = center
	X, Y = np.ogrid[:h, :w]
	a = np.arctan2(y - cy, x - cx) * a180 / np.pi
	A = np.arctan2(Y - cy, X - cx) * a180 / np.pi
	REZ = A - a + 180
	print(w, h, cx,cy,x,y, a)
	return REZ

def create_circle_mask(h, w, radius, center=None):
	if center is None: # use the middle of the image
		center = (int(h/2), int(w/2))
	else:
		center = center[1], center[0]

	X, Y = np.ogrid[:h, :w]
	dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
	mask = dist_from_center <= radius
	return mask

def rectangle(img, pt1, pt2, border=2, color=[0]):
	"""
		img: Input image where we want to draw rectangle:
		pt1: top left point (y, x)
		pt2: bottom right point
		border: border of line
		color: color of rectangle line,
		returns new image with rectangle.
		
	"""
	p1 = pt1
	pt1 = (p1[1], p1[0])
	p2 = pt2
	pt2 = (p2[1], p2[0])
	b = int(np.ceil(border/2))
	cvalue = np.array(color)
	if border >= 0:
		# get x coordinates for each line(top, bottom) of each side
		# if -ve coordinates comes, then make that 0
		x11 = np.clip(pt1[0]-b, 0, pt2[0])
		x12 = np.clip(pt1[0]+b+1, 0, pt2[0])
		x21 = pt2[0]-b
		x22 = pt2[0]+b+1

		y11 = np.clip(pt1[1]-b, 0, pt2[1])            
		y12 = np.clip(pt1[1]+b+1, 0, pt2[1])   
		y21 = pt2[1]-b
		y22 = pt2[1]+b+1
		# right line
		img[x11:x22, y11:y12] = cvalue
		#left line
		img[x11:x22, y21:y22] = cvalue
		# top line
		img[x11:x12, y11:y22] = cvalue
		# bottom line
		img[x21:x22, y11:y22] = cvalue
		
	else:
		pt1 = np.clip(pt1, 0, pt2)
		img[pt1[0]:pt2[0]+1, pt1[1]:pt2[1]+1] = cvalue
		
	return img

def ellipse(img=None, center=(0, 0), a=3, b=1, border=4, color=[0], smooth=2):
	"""
		A method to create a ellipse on a give image.
		img: Expects numpy ndarray of image. 
		center: center of a ellipse
		a: major axis
		b: minor axis
		border: border of the ellipse, if -ve, ellipse is filled
		color: color for ellipse
		smooth: how smooth should our ellipse be?(smooth * 360 angles in 0 to 360)
	"""
	if type(img) == None:
		raise ValueError("Image can not be None. Provide numpy array instead.")
	angles = 360
	cvalue = np.array(color)
	if type(img) != type(None):
		shape = img.shape
		if len(shape) == 3:
			row, col, channels = shape
		else:
			row, col = shape
			channels = 1
		angles = np.linspace(0, 360, 360*smooth)
		for i in angles:
			angle = i*np.pi/180
			y = center[1]+b*np.sin(angle) 
			x = center[0]+a*np.cos(angle)
			# since we are wroking on image, coordinate starts from (0, 0) onwards and we have to ignore -ve values
			if border >= 0:
				r, c = int(x), int(y)
				bord = int(np.ceil(border/2))
				x1 = np.clip(x-bord, 0, img.shape[0]).astype(np.int32)
				y1 = np.clip(y-bord, 0, img.shape[1]).astype(np.int32)
				x2 = np.clip(x+bord, 0, img.shape[0]).astype(np.int32)
				y2 = np.clip(y+bord, 0, img.shape[1]).astype(np.int32)
				
				img[x1:x2, y1:y2] = cvalue
				
			else:
				x = np.clip(x, 0, img.shape[0])
				y = np.clip(y, 0, img.shape[1])
				r, c = int(x), int(y)
				if i > 270:
					img[center[0]:r, c:center[1]] = cvalue
				elif i > 180:
					img[r:center[0], c:center[1]] = cvalue
				elif i > 90:
					img[r:center[0], center[1]:c] = cvalue
				elif i > 0:
					img[center[0]:r, center[1]:c] = cvalue
					 
		return img
