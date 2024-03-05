from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.progressbar import ProgressBar
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from imagevideo import ImageVideo
"""
{
	"cartoon_type":"SlidingDoor",
	"options":{
		"seconds":1,
		"direct":"l-r",
		"old_imfile":"d:\\2017-09-24\\IMG_5398.JPG",
		"imfile":'d:\\2014-01-26 厦门\\IMG_2555.JPG'
	}
},
{
	"cartoon_type":"SlidingDoor",
	"options":{
		"seconds":1,
		"imfile":"d:\\2017-09-24\\IMG_5398.JPG",
		"old_imfile":'d:\\2014-01-26 厦门\\IMG_2555.JPG'
	}
},
{
	"cartoon_type":"CircleIn",
	"options":{
		"seconds":1,
		"imfile":'d:\\2014-01-26 厦门\\IMG_2555.JPG',
		"old_imfile":"d:\\2008-01-26\\IMG_6545.JPG",
		"radius":10
	}
},
{
	"cartoon_type":"Arcs",
	"options":{
		"seconds":1,
		"imfile":'d:\\2014-01-26 厦门\\IMG_2555.JPG',
		"old_imfile":"d:\\2008-01-26\\IMG_6545.JPG"
	}
},
			{
				"cartoon_type":"FadeIn",
				"options":{
					"seconds":1,
					"imfile":"d:\\2008-01-26\\IMG_6545.JPG"
				}
			},
			{
				"cartoon_type":"MagnifierOut",
				"options":{
					"seconds":1,
					"imfile":"d:\\2017-09-24\\IMG_5398.JPG",
					"box":[0,0,400, 300]
				}
			},
			{
				"cartoon_type":"MagnifierIn",
				"options":{
					"seconds":1,
					"imfile":"d:\\2017-09-24\\IMG_5398.JPG",
					"box":[400,400,900, 1000]
				}
			},
			{
				"cartoon_type":"SwitchSmooth",
				"options":{
					"seconds":1,
					"old_imfile":"d:\\2008-01-26\\IMG_6545.JPG",
					"imfile":"d:\\2017-09-24\\IMG_5395.JPG"
				}
			},
			{
				"cartoon_type":"FadeOut",
				"options":{
					"seconds":1,
					"imfile":"d:\\2017-09-24\\IMG_5395.JPG"
				}
			}
"""

class MoveImageApp(App):
	def build(self):
		cartoons = [
			{
				"cartoon_type":"Arcs",
				"options":{
					"seconds":1,
					"clockwise":True,
					"center":(0,0),
					"imfile":'d:\\2014-01-26 厦门\\IMG_2555.JPG',
					"old_imfile":"D:\\2014-01-26 厦门\\IMG_2558.JPG"
				}
			},
			{
				"cartoon_type":"Arcs",
				"options":{
					"seconds":1,
					"clockwise":False,
					"imfile":"D:\\2014-01-26 厦门\\IMG_2575.JPG",
					"old_imfile":'d:\\2014-01-26 厦门\\IMG_2555.JPG'
				}
			}
		]
		box = BoxLayout(orientation='vertical')
		box1 = BoxLayout(orientation='horizontal', height=60, size_hint_y=None)
		x = ImageVideo(repeat=True, cartoons=cartoons)
		x.bind(on_write_progress=self.show_write_progress)
		x.bind(on_write_finish=self.show_write_finished)
		b1 = Button(text='start', size_hint_x=None, width="100")
		b1.bind(on_press=x.start)
		b2 = Button(text='stop', size_hint_x=None, width="100")
		b2.bind(on_press=x.stop)
		b3 = Button(text='save', size_hint_x=None, width="100")
		b3.bind(on_press=x.save)
		self.infobox = BoxLayout(orientation='horizontal')
		
		box.add_widget(box1)
		box.add_widget(x)
		box1.add_widget(b1)
		box1.add_widget(b2)
		box1.add_widget(b3)
		box1.add_widget(self.infobox)
		return box

	def show_write_progress(self, o, d):
		cur_txt_w = Label(text=str(d['current']), size_hint_x=None,
								width=60)
		progressbar = ProgressBar(value=d['rate'])
		total_txt_w = Label(text=str(d['total']), size_hint_x=None,
								width=60)
		self.infobox.add_widget(cur_txt_w)
		self.infobox.add_widget(progressbar)
		self.infobox.add_widget(total_txt_w)
		print('progress:',d)

	def show_write_finished(self, o, d=None):
		print('write finish')
		self.infobox.clear_widgets()

if __name__ == '__main__':
	MoveImageApp().run()

