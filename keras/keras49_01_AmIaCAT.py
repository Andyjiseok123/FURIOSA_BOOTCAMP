from keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score
from keras.preprocessing.image import ImageDataGenerator

np_path = './_data/kaggle_cat_dog_npy/'
my_img = np.load(np_path+ 'keras48_me.npy')
# print(my_img)
my_img = my_img/255.
# print(my_img)

# my_img = my_img.flow_from_directory(
#     my_img,                 #경로
#     target_size=(100,100),
#     batch_size=160,                      
#     class_mode='binary',            #이진분류
#     color_mode='rgb',
#     shuffle=True,
#     )


path = './_save/keras45/'  
# model.save_weights(path + 'keras29_5_save_weights1.weights.h5') #가중치 세이브
# model.save(path + 'keras45_save_model_cat_dog.keras') #모델 세이브
model = load_model(path + 'keras45_save_model_cat_dog.keras') #저장된 모델 불러오기

img_predict = model.predict(my_img)
img_predict = np.round(img_predict)
if img_predict == 0:
    print('나는 고양이')
else:
    print('나는 개')