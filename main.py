from deepface import DeepFace 
import matplotlib.pyplot as plt
import cv2


img1_path = 'images/luca.jpg'
img2_path = 'images/luca2.jpg'

models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib', 'ArcFace', 'Ensemble']  #Ensemble is a combination of all the models, its super accurate but slow AF...not recommended
model_name = models[0]

resp = DeepFace.verify(img1_path = img1_path, img2_path = img2_path, model_name = model_name)

# print(resp)

if resp['verified'] == True:
    print('Same person')
    
else:
    print('Different person')
    
#see the face detection in action

img1 = DeepFace.detectFace(img1_path)
img2 = DeepFace.detectFace(img2_path)

img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

cv2.imshow("Image", img1)
cv2.imshow("Image2", img2)

cv2.waitKey(0)