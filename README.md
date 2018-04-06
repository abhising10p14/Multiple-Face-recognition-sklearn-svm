# Face Recognition using Sklearn Svm and OpenCv

This repository deals with enhacing the accuracy of face recognition method. Generally, on the internet, you would find everyone or most of the people using either the haarcascade, eigenface models provided by the openCv or the neuralnetwork models(Facenet,Nnet e.t.c) for face recognition. Even I have used the OpenCv haarcascade[https://github.com/abhising10p14/Face-Recognition-LBPH]. There are two shortcomes of both the approaches:

	1.In case of the Opencv haarcascade and the fischerface, the accuracy was not so good (60-70%)
	2.Incase of CNN neural network, a huge amount of processing power is required, thoug the accuracy was far better than the OPencv models

To overcome these shortcomes, SVM along with OpenCV(to detect the faces) is used. This gives a descent accuracy of () as well as requires less processing time.

## What is the algorithm?

In this repostory SVM is used to train the model using the training images(**orl_faces**). The sklearn inbuilt svm classifier extracts the features from the given training images in the orl_faces folder. Later on, The test images are passed in the test-data folder on which testing is done.

## Requirements
	OpenCv
	Sklearn
	Scipy

## How to use?

1. First of all put you training datset into the folder **orl_faces/data**. You can see the format in which images have been put in the corresponding folders. The name of the subfolder should be the name you want to predict when passing an image. Make sure you provide a good data set. The data set should have proper frontal faces for a better accuracy.
2. Delete the current images present in the **output/data** folder.
3. Now put your testing images into the folder **test-data/data** folder.
4. Run the command : **python conversion.py**  This will convert all your training as well as the testing images 
	into the greyscale format supported by the svm classifier. Currently , **.jpeg, .jpg, .png** format is supported. If you want to add any other type of format then correspondingly change the code of conversion.py
5. Now run  **crop_faces.py**  This finds the faces in your test data and removes the rest part of the 
	picture. for detecting the faces, haarcascad_frontal_faces.xml and eye_.xml have been used. These models are present in the **models** folder. After running this command check the subfolders of the orl_faces folder that there are not any such image in the subfolders where other parts of pictures other than the faces are present. If Found, delete them because they are going to affect your training model. You can add this as condition check in the crop_faces.py 
5. Now run **multiple_faces.py**  This finds the faces in your test images. If faces are found, it saves 
	the corresponding faces in the **output/data** folder. After runnig this command a new file **dictionary.txt** will be generated. This file maps the detected face to the original images of the **test-data/data**. This file is used to display the predicted names on the corresponding testing images.
6. Now run  **svm.py** You need to run this command twice. First time, for training the model over the   
	orl_faces and saving the model as **finalized_model.sav**. Second time, for the purpose of testing your model over the test-data.
  
 The accuracy over the current data set is 85.4% which can vary. To enhance the accuracy use proper good quality frontal face images. Accuaracy increase with increase in the sample size of the dataset.

 For confusion, error, suggestion mail me at[abhisingh10p14@gmail.com]