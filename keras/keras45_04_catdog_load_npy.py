import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.metrics import accuracy_score

#1. 데이터

np_path = './_data/kaggle_cat_dog_npy/'
x_train = np.load(np_path+'keras45_01_x_train.npy')
y_train = np.load(np_path+'keras45_01_y_train.npy')
x_test = np.load(np_path+'keras45_01_x_test.npy')
y_test = np.load(np_path+'keras45_01_y_test.npy')

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

# exit()


#2. 모델 구성
model = Sequential()
model.add(Conv2D(32,(5,5),padding='same', activation='relu',input_shape = (100,100,3)))
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Conv2D(16,(5,5),padding='same',activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Conv2D(8,(5,5),padding='same',activation='relu'))
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
          epochs = 2000, batch_size = 400,
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


# batch_size = 5000
# accuraccy_score:  0.7993079584775087
# 소요시간 :  731.83 초

# batch_size = 10000
# accuraccy_score:  0.7834898665348492
# 소요시간 :  1987.4 초