import time
from io import BytesIO
from PIL import Image as PILImage
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from kivy.properties import StringProperty, BooleanProperty, \
			NumericProperty, ListProperty
from appPublic.background import Background
from kivyblocks.utils import blocksttf
from image_cartoon import cartoonBuild
from to_video import write_video, save_as_video
from image_text import Subtitles

subtitles = [
	{
		"start_at":0,
		"end_at":2,
		"text":"this at 1 second"
	},
	{
		"start_at":2,
		"end_at":3.1,
		"text":"this at 2 second"
	}
]

class ImageVideo(Image):
	repeat = BooleanProperty(False)
	fps = NumericProperty(120)
	cartoons = ListProperty([])
	write_cnt = NumericProperty(0)
	def __init__(self, **kw):
		self.frames = []
		self.cur_cnt = 0
		self.frame_cnt = len(self.frames)
		self.show_task = None
		self.total_frames = 0
		self.size_task = None
		super().__init__(**kw)
		font_file=blocksttf('DroidSansFallback.ttf')
		self.register_event_type('on_write_progress')
		self.register_event_type('on_write_finish')
		self.subtitle = Subtitles(subtitles, font_file, 
							color=(255,255,255), 
							font_size=32)

	def on_write_finish(self, *args):
		pass

	def on_write_progress(self, *args):
		"""
		d format
		{
			"total":x,
			"current":y,
			"rate": z
		}
		"""
		pass

	def save(self, *args):
		b = Background(self._save)
		b.start()

	def _save(self):
		size = (1920, 1080)
		fps=120
		[ f.change_size(size) for f in self.frames ]
		[ f.set_fps(fps) for f in self.frames ]
		self.save_video(fps=fps, size=(1920, 1080))
		[ f.change_size(self.size) for f in self.frames ]
		[ f.set_fps(self.fps) for f in self.frames ]

	def save_video(self, outfile='video1.mp4', fps=None, size=None):
		if size is None:
			size = self.size
		if fps is None:
			fps = self.fps

		total_frames = self.calc_total_frames(fps)
		bcnt = len(str(total_frames))
		img_fmt = 'img_%%0%dd.jpeg' % bcnt
		print(img_fmt)
		outpath='./tmp'
		save_as_video(self, outpath, outfile, './mylogo.png', img_fmt, 
					img_size=size,
					fps=fps)
		

	def calc_total_frames(self, fps):
		total_frames = 0
		for f in self.frames:
			total_frames += f.seconds * fps
		return total_frames

	def on_cartoons(self, *args):
		self.frames = []
		for c in self.cartoons:
			self.frames.append(cartoonBuild(c))
		self.frame_cnt = len(self.frames)
		self.change_frame_size()

	def on_size(self, *args):
		if self.size_task:
			self.size_task.cancel()
		self.size_task = Clock.schedule_once(self.change_frame_size,0.1)

	def change_frame_size(self, *args):
		[ f.change_size(self.size) for f in self.frames ]

	def insert_frame(self, index, frame):
		frame.change_size(self.size)
		self.frames.insert(index, frame)

	def append_frame(self, frame):
		frame.change_size(self.size)
		self.frames.append(frame)

	def __iter__(self):
		return self

	def get_frame(self, frame_type='bytes'):
		c = 0
		t = len(self.frames)
		for f in self.frames:
			r = True
			while r:
				try:
					img = next(f)
					if frame_type == 'bytes':
						yield img.tobytes()
					else:
						yield img

				except StopIteration:
					r = False
			print(t, c, 'switch to next')
			c += 1
		
	def show_image_task(self, *args):
		try:
			tim1 = time.time()
			f = self.frames[self.cur_cnt]
			img = next(f)
			self.show_image(img)
			self.time_pass += self.time_period
			tim2 = time.time()
			print(f.__class__.__name__, 'frame time cost=', tim2 - tim1)
		except StopIteration:
			self.cur_cnt += 1
			if self.cur_cnt >= self.frame_cnt:
				if self.repeat:
					self.initial()
				else:
					self.stop()
					return
			self.show_image_task()

	def show_image(self, img):
		img = self.subtitle.subtitle(self.time_pass, img)
		w, h = img.size
		nh = 0
		nw = 0
		if (self.width / self.height) > (w / h):
			nh = self.height
			nw = w / h * nh
		else:
			nw = self.width
			nh = h / w * nw
		if nw <= 0 or nh <= 0:
			return
		img.resize((int(nw), int(nh)))
		data = BytesIO()
		img.save(data, format='jpeg')
		data.seek(0)
		im = CoreImage(data, ext='jpeg')
		self.texture = im.texture

	def on_fps(self, *args):
		[ f.set_fps(self.fps) for f in self.frames ]

	def initial(self):
		self.cur_cnt = 0
		for f in self.frames:
			f.initial()

	def start(self, *args):
		self.initial()
		self.on_size()
		self.on_fps()
		self.time_pass = 0
		self.time_period = 1 / self.fps
		self.show_task = Clock.schedule_interval(self.show_image_task, \
									1/self.fps)

	def stop(self, *args):
		self.show_task.cancel()
		self.show_task = None
