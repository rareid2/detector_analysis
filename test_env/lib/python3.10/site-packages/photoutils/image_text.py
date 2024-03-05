from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

def textsplit(txt):
	ret = []
	t = ''
	for w in txt:
		if ord(w) > 255:
			ret.append(w)
			continue
		if w == ' ':
			if t != '':
				ret.append(t)
			t = ''
			continue
		t = f'{t}{w}'
	if t != '':
		ret.append(t)
	return ret

def textjoin(words):
	i = 0
	ret = ''
	words_len = len(words) - 1
	for i, w in enumerate(words):
		ret = f'{ret}{w}'
		if (ord(w[0]) < 256 or ord(words[i][0]) < 256) and i < words_len:
			ret = f'{ret} '
	return ret
		
class ImageText:
	def __init__(self, font_file, location='bottom', margin=10):
		self.font_file = font_file
		self.location = location
		self.margin = margin
	
	def get_text_box(self, text, font_size, 
							font_file=None, maxwidth=None):
		"""
		return box size, and textarray
		"""
		if maxwidth is None:
			return self.get_text_size(text,font_size, font_file=font_file), [text]
		words = textsplit(text)
		b_i = 0
		texts = []
		width = 0
		height = 0
		words_len = len(words) - 1
		for i, w in enumerate(words):
			t = textjoin(words[b_i:i])
			w, h = self.get_text_size(t, font_size, font_file=font_file)
			if w > maxwidth:
				l = textjoin(words[b_i:i-1])
				texts.append(l)
				b_i = i - 1
				height += h
		l = textjoin(words[b_i:])
		texts.append(l)
		return (maxwidth, height), texts
		 
	def get_text_size(self, text, font_size, font_file=None):
		if font_file is None:
			font_file = self.font_file
		print('font_file=', font_file, 'font-size=', font_size)
		font = ImageFont.truetype(font_file, font_size)
		return font.getsize(text)

	def text_to_image(self, text, font_size, color, max_width=None, font_file=None, ):
		if font_file is None:
			font_file = self.font_file
		size, txts = self.get_text_box(text, font_size, 
							file_file=font_file, maxwidth=max_width)

		font = ImageFont.truetype(self.font_file, font_size)
		img = Image.new("RGB", size, (0,0,0))
		drawer = ImageDraw.Draw(img)
		for txt in txts:
			drawer.text((0, h), txt, color, font=font)
			w, h1 = self.get_text_size(txt, font_size, font_file=font_file)
			h += h1
		return img

	def text(self, img, text, color, font_size=None):
		iw, ih = img.size
		if font_size is None:
			font_size = int(iw / 20)
		pos = iw * 0.15 / 2
		size, txts = self.get_text_box(text, font_size, 
							maxwidth=iw * 0.85)

		font = ImageFont.truetype(self.font_file, font_size)
		drawer = ImageDraw.Draw(img)
		x = pos
		y = ih - pos - size[1]
		for txt in txts:
			drawer.text((x, y), txt, color, font=font)
			w, h = self.get_text_size(txt, font_size)
			y += h
		return img

class Subtitles:
	def __init__(self, subtitles, font_file, 
						location='bottom',
						margin=10,
						color=(255, 255, 255), 
						font_size=24):
		self.subtitles = subtitles
		self.i_t = ImageText(font_file, location=location,margin=margin)
		self.color = color
		self.font_size =font_size
		self.max_cnt = len(subtitles)
		self.cur_pos = 0

	
	def subtitle(self, tim, img):
		if self.cur_pos >= self.max_cnt:
			return img

		st = self.subtitles[self.cur_pos]
		while st['end_at'] <= tim:
			self.cur_pos += 1
			if self.cur_pos >= self.max_cnt:
				return img
			st = self.subtitles[self.cur_pos]

		if tim >= st['start_at'] and tim < st['end_at']:
			self.i_t.text(img, st['text'], self.color)
		
		return img
