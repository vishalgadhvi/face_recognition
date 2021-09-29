# Import OpenCV2 for image processing
import cv2
import os
from shutil import copyfile
import glob

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

path = 'D:\\College Work\\6th sem\\SGP2\\MultiDataSet\\'
filenames=os.listdir(path)
L = len(filenames)

# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()

assure_path_exists("trainer/")

# Load the trained mode
recognizer.read('trainer/trainer.yml')

Id=0
# Load prebuilt model for Frontal Face
cascadePath = "haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX
print('Running...')
for i in range(L):
    filename=filenames[i]
    im = cv2.imread('D:\\College Work\\6th sem\\SGP2\\MultiDataSet\\' + filename,1)    #cam.read()#
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    # Get all face from the video frame
    faces = faceCascade.detectMultiScale(gray, 1.2,5)

    # For each face in faces
    for(x,y,w,h) in faces:

            # Create rectangle around the face
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

            # Recognize the face belongs to which ID
        Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])  

            # Check the ID if exist

        if(Id == 1):
            copyfile('D:\\College Work\\6th sem\\SGP2\\MultiDataSet\\' + filename, 'D:\\College Work\\6th sem\\SGP2\\Manthan\\' + filename)
            Id = "Manthan"

        if(Id == 2):
            copyfile('D:\\College Work\\6th sem\\SGP2\\MultiDataSet\\'+ filename, 'D:\\College Work\\6th sem\\SGP2\\Nirav\\' + filename)
            Id = "Nirav"

        if(Id == 3):
            Id="Smit"

print('Finished')
