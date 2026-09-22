# 50-1 copy
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping

(x_train, y_train),(x_test,y_test) = mnist.load_data()
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)
# exit()


datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    # height_shift_range=0.1,
    rotation_range=5,
    # zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)

augment_size = 40000

randidx = np.random.choice(x_train.shape[0], size=augment_size)
print(randidx)
print(randidx.shape)
print(len(randidx))
# print(x_train.shape)            #(60000,28,28)
# print(x_train[0].shape)         #(28,28)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

print(x_augmented.shape,y_augmented.shape)      #(40000, 28, 28) (40000,)

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2],1)

x_augmented = datagen.flow(
    x_augmented, y_augmented,
    batch_size = augment_size,
    shuffle = False,
).next()[0]

print(x_augmented.shape)       # (40000, 28, 28, 1)

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

x_train =  np.concatenate((x_train,x_augmented))
y_train =  np.concatenate((y_train,y_augmented))
print(x_train.shape, y_train.shape)                 # (100000, 28, 28, 1) (100000,)
# print(np.unique(y_train,return_counts=True))
# exit()
########################### 스케일링 1. ###########################
x_train = x_train/255.
x_test = x_test/255.

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1,1))
y_test = ohe.fit_transform(y_test.reshape(-1,1))

#2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (3,3), padding='same', input_shape = (28,28,1)))                      #(26,26,64)
model.add(MaxPooling2D())
model.add(Conv2D(filters=64 , kernel_size=(3,3), padding='same', activation='relu'))        #(24,24,32)
model.add(Dropout(0.2))
model.add(Conv2D(32,(3,3), padding='same',activation='relu'))                               #(23,23,32)
model.add(MaxPooling2D())
model.add(Conv2D(32,(3,3), padding='same',activation='relu'))                               #(22,22,16)
model.add(Dropout(0.2))
model.add(Conv2D(32,(2,2), padding='same',activation='relu'))                               #(21,21,16)
model.add(Dropout(0.2))
model.add(Conv2D(32,(2,2), padding='same',activation='relu'))                               #(20,20,16)

model.add(GlobalAveragePooling2D())                                                        #이후 FC layer와 붙기 위해 한줄로 reshape

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))    

# model.summary()
# exit()

#3. 컴파일, 훈련
import time
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics = ['acc'])
start_time = time.time()
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=100,
                   restore_best_weights=True)
model.fit(x_train,y_train,
          epochs = 2000, batch_size = 1000,
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
y_predict = np.argmax(y_predict, axis=1).reshape(-1,1)
y_test = np.argmax(y_test,axis=1).reshape(-1,1)
from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test,y_predict)
print('accuraccy_score: ',acc_score)
print('소요시간 : ',round(end_time-start_time,2), '초')


# accuraccy_score:  0.9144
# 소요시간 :  1788.31 초