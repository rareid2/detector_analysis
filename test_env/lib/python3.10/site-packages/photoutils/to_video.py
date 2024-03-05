import os
import subprocess
#import sh
from appPublic.folderUtils import _mkdir
from appPublic.folderUtils import listFile
from ffpyplayer.writer import MediaWriter
from ffpyplayer.pic import Image as FFImage

def save_as_video(source, outpath, outfile, log_img_path, img_fmt, 
					img_size=(640, 480),
					fps=120, meta={}):
	_mkdir(outpath)
	i = 1
	for im in source.get_frame(frame_type='image'):
		f = img_fmt % i
		print(f)
		fn = os.path.join(outpath, f)
		im.save(fn)
		i += 1

	w, h = img_size
	if os.path.exists(outfile):
		os.remove(outfile)
	print(f'ffmpeg -r {fps} -f image2 -s {w}x{h} -i {outpath}/{img_fmt} -i {log_img_path} -filter_complex [0:v][1:v] overlay=0:0 -vcodec libx264 -crf 15 -pix_fmt yuv420p {outfile}')
	subprocess.run(f'ffmpeg -r {fps} -f image2 -s {w}x{h} -i {outpath}/{img_fmt} -vcodec libx264 -crf 15 -pix_fmt yuv420p {outfile}')
	"""
	delete logo image
	sh.ffmpeg("-r", fps, 
				"-f", "image2",
				"-s", f"{w}x{h}",
				"-i", f"{outpath}/{img_fmt}",
				"-i", log_img_path, "-filter_complex", "[0:v][1:v] overlay=0:0",
				"-vcodec", "libx264",
				"-crf", "15",
				"-pix_fmt", "yuv420p",
				outfile)
	"""

	for f in listFile(outpath):
		os.remove(f)
	os.rmdir(outpath)
	print(f'{outfile} file created')

def write_video(source, outfile=None,
					img_size=(600, 400),
					fps=120,
					meta={}
				):

	w, h = img_size
	out_opts = {
		'pix_fmt_in':'rgb24',
		'width_in':w,
		'height_in':h,
		'codec':'libx264',
		'frame_rate':(120,1)
	}
	lib_opts = {
		'preset':'slow',
		'crf':'22'
	}
	metadata = meta
	writer = MediaWriter(outfile,
					[out_opts]*2, fmt='mp4',
					width_out=w, height_out=h,
					pix_fmt_out='yuv420p',
					lib_opts=lib_opts, metadata=metadata)
	p = 1 / fps
	i = 0
	for im_bytes in source.get_frame():
		yield i
		im_ba = bytearray(im_bytes)
		im = FFImage(plane_buffer=[im_ba], pix_fmt='rgb24', size=(w,h))
		pts = p * i
		r = writer.write_frame(im, pts)
		i += 1
	writer.close()
	print('Write_finished')


