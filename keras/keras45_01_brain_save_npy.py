import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.metrics import accuracy_score

#1. 데이터

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    # horizontal_flip = True,              #수평반전
    # vertical_flip= True,                #수직반전
    # width_shift_range= 0.1,             #평형이동
    # height_shift_range= 0.1,            #수직이동
    # rotation_range=5,                   #각도조절
    # zoom_range=1.2,                     #확대축소
    # shear_range=0.7,                    #전단변형
    # fill_mode='nearest',                #채우기
    )

test_datagen = ImageDataGenerator(
    rescale=1./255,
    )

path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(
    path_train,                 #경로
    target_size=(100,100),
    batch_size=160,                      
    class_mode='binary',            #이진분류
    color_mode='grayscale',
    shuffle=True,
    )

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(100,100),
    batch_size=120,                      
    class_mode='binary',            #이진분류
    color_mode='grayscale',
    shuffle=False,             # test에서는 필요가 없다
)

x_train = xy_train[0][0]
y_train = xy_train[0][1]

x_test = xy_test[0][0]
y_test = xy_test[0][1]

np_path = './_data/kaggle_cat_dog_npy/'
np.save(np_path+'keras45_01_x_train.npy', arr=x_train)
np.save(np_path+'keras45_01_y_train.npy', arr=y_train)
np.save(np_path+'keras45_01_x_test.npy', arr=x_test)
np.save(np_path+'keras45_01_y_test.npy', arr=y_test)

exit()


#2. 모델 구성
model = Sequential()
model.add(Conv2D(160,(5,5),padding='same', activation='relu',input_shape = (100,100,1)))
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Conv2D(80,(5,5),padding='same',activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Conv2D(20,(5,5),padding='same',activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(GlobalAveragePooling2D())
model.add(Dense(16,activation = 'relu'))
model.add(Dense(8,activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))
# model.summary()
# exit()

#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
              metrics = ['acc'])
start_time = time.time()
es = EarlyStopping(monitor='val_acc',
                   mode='max',
                   patience=100,
                   restore_best_weights=True)
model.fit(x_train,y_train,
          epochs = 2000, batch_size = 20,
          verbose = 1,
          validation_split = 0.3,
          callbacks = [es])
end_time = time.time()


#4. 평가, 예측

print("=======================model.evaluate===========================")
loss = model.evaluate(x_test,y_test,
                     verbose = 1)
print('loss: ',loss[0])
print('acc: ',loss[1])

y_predict = model.predict(x_test)
# y_predict = np.argmax(y_predict, axis=1).reshape(-1,1)
# y_test = np.argmax(y_test,axis=1).reshape(-1,1)

acc_score = accuracy_score(y_test,np.round(y_predict))
print('accuraccy_score: ',acc_score)
print('소요시간 : ',round(end_time-start_time,2), '초')

# accuraccy_score:  0.9416666666666667
# 소요시간 :  135.48 초