from deepface import DeepFace 
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import face_recognition

image = 'imagetest/angelica.jpg'
model_name = "Facenet"
#store results in a dataframe, to see the results of all image comparings
df = DeepFace.find(img_path = image, db_path = 'images', model_name = model_name)


print(df.head())
if not df.empty:
    message = "Authorized"
    print("FACE IS FOUND IN THE DATABASE")
    
else:
    message = "Unknown"
    print("FACE NOT FOUND")
    
# print(df.head())

img = cv2.imread(image)

imgS = cv2.resize(img,(0,0),None,0.25,0.25) #compress the image to improve performance
imgS = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 

facesCurFrame = face_recognition.face_locations(imgS)

for faceLoc in facesCurFrame:
    
    if message == "Authorized":
        y1,x1,y2,x2 = faceLoc
        # y1,x1,y2,x2 = y1*4,x1*4,y2*4,x2*4 
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(img, (x1,y2-35),(x2,y2),(0,255,0), cv2.FILLED) #starting point on height reduced by -35 to be a little lower so we can write the name on top of this rectangle
        cv2.putText(img,message, (x2, y2-6), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        
    else:
        y1,x1,y2,x2 = faceLoc
        # y1,x1,y2,x2 = y1*4,x1*4,y2*4,x2*4 
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.rectangle(img, (x1,y2-35),(x2,y2),(0,0,255), cv2.FILLED) #starting point on height reduced by -35 to be a little lower so we can write the name on top of this rectangle
        cv2.putText(img,message, (x2, y2-6), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    

h, w = img.shape[:2]
h = int(h/2)
w = int(w/2)
img = cv2.resize(img, (h,w))
cv2.imshow("Image", img)
cv2.waitKey(0)