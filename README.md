#Face Recognition using Sklearn Svm and OpenCv

This repository deals with enhacing the accuracy of face recognition method. Generally, on the internet, you would find everyone or most of the people using either the haarcascade, eigenface models provided by the openCv or the neuralnetwork models(Facenet,Nnet e.t.c) for face recognition. Even I have used the OpenCv haarcascade[https://github.com/abhising10p14/Face-Recognition-LBPH]. There are two shortcomes of both the approaches:

	1.In case of the Opencv haarcascade and the fischerface, the accuracy was not so good (60-70%)
	2.Incase of CNN neural network, a huge amount of processing power is required, thoug the accuracy was far better than the OPencv models

To overcome these shortcomes, SVM along with OpenCV(to detect the faces) is used. This gives a descent accuracy of () as well as requires less processing time.

##What is the algorithm?

In this repostory SVM is used to train the model using the training images(**orl_faces**). The sklearn inbuilt svm classifier extracts the features from the given training images in the orl_faces folder. Later on, The test images are passed in the test-data folder on which testing is done.

##Requirements
	OpenCv
	Sklearn
	Scipy

##How to use?

1. First of all put you training datset into the folder **orl_faces**. You can see the format in which images have been put in the corresponding folders.