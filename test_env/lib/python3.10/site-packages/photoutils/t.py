import time
from kivyblocks.blocksapp import BlocksApp
from kivy.factory import Factory
from rembg import remove, new_session
from PIL import Image as PILImage
import numpy as np
from appPublic.registerfunction import registerFunction
from mediapipe_remove import MediapipeRembg
# from cvzone_rembg import CVZoneRembg
# from rembg_remove import do_remove

class PhotoApp(BlocksApp):
	def build(self):
		# registerFunction('bg_remove', do_remove)
		_rmbg = MediapipeRembg()
		registerFunction('bg_remove', _rmbg.remove_ff)
		b = Factory.Blocks().widgetBuild
		desc = {
			"widgettype":"VBox",
			"options":{
				"VideoBehavior":{
					"v_src":'S01E01.mp4', 
					"auto_play":True,
					"prehandlers":['bg_remove']
				}
			}
		}

		w = b(desc)
		return w

if __name__ == '__main__':
	PhotoApp().run()
