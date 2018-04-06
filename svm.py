import os
import pdb
import glob
import scipy.misc 
import argparse
import numpy as np
from sklearn import svm,metrics 
from sklearn.cross_validation import ShuffleSplit
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import matplotlib.image as mpimg
import cv2
import config
import math

check = {"test30.jpg":"abhishek","test29.jpg":"AamairKhan","test28.jpg":"AamairKhan","test26.jpg":"AkshayKumar","test25.jpg":"AamairKhan","test27.jpg":"AkshayKumar","1.jpg":"siddhartmalhotra","2.jpg":"AliaBhatt","6.jpg":"varundhawan","test7.jpg":"sharukhkhanAliaBhatt","test24.jpg":"AliaBhattsiddhartmalhotra","test29.jpeg":"AkshayKumar","test30.jpeg":"AkshayKumar","test31.jpeg":"AamairKhan","test32.jpg":"AamairKhan","test33.jpeg":"sharukhkhan","test34.jpeg":"sharukhkhan"}

def load_data(path):
	images = []
	labels = []
	files = glob.glob(os.path.join(path,"*/*"))
	#print files
	for f in files:
		img = scipy.misc.imread(f)
		bc = scipy.misc.imresize(img, (112, 92))
		result = bc
		if(np.array(bc).ndim==3):
			result = bc[:,:,0]
		images.append(result)
		#images.append(bc)
		file_name = f.split("/")[-2]
		labels.append(file_name)
	return np.array(images).reshape((len(images),-1)),np.array(labels)

def load_data_test(path):
	images = []
	labels = []
	files = glob.glob(os.path.join(path,"*/*"))
	files.sort()
	for f in files:
		img = scipy.misc.imread(f)
		bc = scipy.misc.imresize(img, (112, 92))
		result = bc[:,:,0]
		images.append(result)
		file_name = f.split("/")[-1]
		labels.append(file_name)
	
	return np.array(images).reshape((len(images),-1)),np.array(labels)


#runs for testing only
#========================================================================================


def test_it(model,images,valid_labels,verbose=True):
	prediction = model.predict(images)
	if verbose:
		pretty_print_it(prediction,valid_labels)
	return prediction

def pretty_print_it(prediction,valid_labels):
	assert prediction is not None,"Prediction was not performed"
	print ("==============================================")
	#print metrics.classification_report(valid_labels,prediction)
	print ("==============================================")



def calculate_accuracy(test_labels,prediction,dictionary):
	overall_accuracy = 0.0
	total_img = len(prediction)
	test_path = "test-data/data/"
	count = 0
	for img in test_labels:
		name = prediction[count]
		count = count + 1
		crop_test_image_path = "output/data/"+img
		x,y,w,h = dictionary[crop_test_image_path]
		ans = img.split("_")[0]
		ans2 = img.split(".")[-1]
		img =ans + "."+ans2
		if ((check[img] == name) or (check[img].find(name)>=0)) :
			overall_accuracy = overall_accuracy +1
		image_path = test_path  + ans + "."+ans2
		actual_image = cv2.imread(image_path)
		print (image_path)
		cv2.putText(actual_image,name, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1, cv2.LINE_AA)
		#cv2.putText(actual_image,name,(10, 50),cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1, cv2.LINE_AA)
		cv2.imshow("camera", actual_image)
		if cv2.waitKey(800) & 0xFF == ord('q'):
			break
	overall_accuracy = (overall_accuracy/total_img)*100.0
	print ("The overall accuracy is :"  +  str(overall_accuracy))

#=======================================================================================


class SvmClassifier(object):
	def __init__(self,args,gamma=0.001):
		"""
			args: command-line arguments
			gamma : paramter to svm 
	
		"""
		self.args = args
		self.images,self.labels = load_data("orl_faces")
		#print "Loaded images and labels of shape {} and {}".format(self.images.shape,self.labels.shape)
		self.classifier = svm.LinearSVC(verbose=args.verbose)
		self.length = self.images.shape[0]

	def train(self):
		"""
			Train SVM

		"""
		print ("Starting Training")
		
		rs = ShuffleSplit(self.length,n_iter=self.args.fold,test_size=self.args.test_size,random_state=self.args.random_state)
		self.fold = 1
		for train_index,test_index in rs:
			self.train_images,self.train_labels = self.images[train_index,...],self.labels[train_index,...]
			self.valid_images,self.valid_labels = self.images[test_index,...],self.labels[test_index,...]
			#print self.train_labels
			self.svm_classifier = self.classifier.fit(self.train_images,self.train_labels)
			# save the model to disk
			self.fold+=1

		filename = 'finalized_model.sav'
		pickle.dump(self.svm_classifier, open(filename, 'wb'))



if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="SVM for Face Recognition")
	parser.add_argument("-f",dest="foldername",type=str,default="orl_faces")
	parser.add_argument("-fold",dest="fold",type=int,default=5)
	parser.add_argument("-test",dest="test_size",type=float,default=0.25)
	parser.add_argument("-r",dest="random_state",type=int,default=None)
	parser.add_argument("-g",dest="glob_search",type=str,default="*/*")
	parser.add_argument("-v",dest="verbose",type=bool,default=False)
	#parser.add_argument("-k",dest="kernel",type=str,default="rbf")
	args = parser.parse_args()

	
	#svm.display(10)

	model_file = Path("finalized_model.sav")
	dictionary = {}
	if (model_file.is_file()):
		print ("model is already present")
		with open("dictionary.txt","rb") as fp:
			dictionary = pickle.load(fp)
		with open("finalized_model.sav","rb") as fp:
			model = pickle.load(fp)
			test_images,test_labels = load_data_test("output")
			prediction  = test_it(model,test_images,test_labels)
			print ((prediction))
			print ((test_labels))
			calculate_accuracy(test_labels,prediction,dictionary)
			


	else:
		svm = SvmClassifier(args)
		svm.train()
		




