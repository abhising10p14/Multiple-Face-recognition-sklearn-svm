import os
import glob
import cv2


		

def convert_to_grey(path):
	files = glob.glob(os.path.join(path,"*/*"))

	for f in files:
		img = cv2.imread(f)
		img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
		str_ing = []
		if(f.find(".jpg")>=0):
			str_ing = f.split(".jpg")
			final = str_ing[0] + ".jpg"
			cv2.imwrite( final, img )
			print final
		if(f.find(".png")>=0):
			str_ing = f.split(".png")
			final = str_ing[0] + ".png"
			cv2.imwrite( final, img )
			print final
		if(f.find(".jpeg")>=0):
			str_ing = f.split(".jpeg")
			final = str_ing[0] + ".jpeg"
			cv2.imwrite( final, img )
			print 
		if(f.find(".pgm")>=0):
			continue
		
		
		
		

convert_to_grey("orl_faces")
convert_to_grey("test-data")
