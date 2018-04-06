import config
import numpy as np
import cv2
import os, os.path
import glob
import pickle
#### the counter
cnt = 0
face_cascade = cv2.CascadeClassifier(config.FACE_CASCADE_FILE)
eye_cascade = cv2.CascadeClassifier(config.EYE_CASCADE_FILE)

DIR = 'test-data'
numPics = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
dictionary = {}

files = glob.glob(os.path.join(DIR,"*/*"))

for f in files:
    
    img = cv2.imread(f)
    height = img.shape[0]
    width = img.shape[1]
    size = height * width
    temp_img = img
    if size > (500^2):
        r = 500.0 / img.shape[1]
        dim = (500, int(img.shape[0] * r))
        img2 = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        img = img2

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    cas_rejectLevel = 1.3
    cas_levelWeight = 5
    pic = 1
    for l in faces:
        #print l
        eyesn = 0
        x = l[0]
        y = l[1]
        w = l[2]
        h = l[3]
        temp_list = (x,y,w,h)
        imgCrop = img[y:y+h,x:x+w]
        cv2.rectangle(temp_img,(x,y),(x+w,y+h),(255,0,0),3)
        cv2.putText(temp_img," face", (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2, cv2.LINE_AA)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            eyesn = eyesn +1
        if eyesn >= 2:
            #### increase the counter and save 
            file_name = f.split("/")[-1]
            check = file_name.split(".")[0]
            check_formaat = file_name.split(".")[-1]
            ans = "output/data/" + check + "_"+ str(pic)+"." +check_formaat
            cv2.imwrite(ans, imgCrop)
            temp = "output/data/" + check +"_"+ str(pic)+ "."+ check_formaat
            pic = pic+1
            temp_dict = {temp:temp_list}
            dictionary.update(temp_dict)
            cv2.imshow('img',temp_img)
            #print("Image"+str(pic)+" has been processed and cropped")
            if cv2.waitKey(200) & 0xFF == ord('q'):
                break
    

#cap.release()
pickle.dump(dictionary, open("dictionary.txt", 'wb'))
print("All images have been processed!!!")
cv2.destroyAllWindows()
cv2.destroyAllWindows()