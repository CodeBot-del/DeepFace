from deepface import DeepFace 
import matplotlib.pyplot as plt
import cv2

img1_path = 'images/assembly.jpg'

models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib', 'ArcFace', 'Ensemble']  #Ensemble is a combination of all the models, its super accurate but slow AF...not recommended
model_name = models[0]

img1 = DeepFace.detectFace(img1_path)
img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
cv2.imshow("Image", img1)
cv2.waitKey(0)



    