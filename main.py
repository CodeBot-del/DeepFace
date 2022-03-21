from deepface import DeepFace 

img1_path = 'images/yara1.jpg'
img2_path = 'images/luca.jpg'

model_name = 'Facenet'

resp = DeepFace.verify(img1_path = img1_path, img2_path = img2_path, model_name = model_name)

# print(resp)

if resp['verified'] == True:
    print('Same person')
    
else:
    print('Different person')
