from PIL import Image
from image_text import ImageText

font_file='/home/ymq/py/kivyblocks/kivyblocks/ttf/DroidSansFallback.ttf'

it = ImageText(font_file=font_file)
text = "Bitmap fonts are stored"
# text = "Bitmap fonts are stored in PIL’s own format, where each font typically consists of two files, one named .pil and the other usually named .pbm. The former contains font metrics, the latter raster data."
im_file="/home/ymq/c/images/2014-01-26 厦门/IMG_2555.JPG"

im = Image.open(im_file)
it.text(im, text, (255,0,0))
im.save('./t.jpg')
