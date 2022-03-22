from deepface import DeepFace 
import matplotlib.pyplot as plt
import cv2
import pandas as pd

#find function returns whether the face is found or not found. 
#it takes two basic arguments... the image to be compared, and the path to the database, other arguments like the model, face detector, and so on can also be defined

image = 'imagetest/luka2.jpeg'
#store results in a dataframe, to see the results of all image comparings
df = DeepFace.find(img_path = image, db_path = 'images')

# print(df.head())

#Analyze the face (you can pass actions to specify analysis, actions = ['emotion', 'age', 'gender', 'race') #without specifying, the function returns both actions.

obj = DeepFace.analyze(img_path= image)
print(obj)             

