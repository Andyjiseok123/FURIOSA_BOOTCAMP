import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

#1. 데이터

np_path = './_data/gender_npy/'
x_train = np.load(np_path+'keras49_01_x_train.npy')
x_test = np.load(np_path+'keras49_01_x_test.npy')
y_train = np.load(np_path+'keras49_01_y_train.npy')
y_test = np.load(np_path+'keras49_01_y_test.npy')

# print(x_train.shape)        #(23091, 100, 100, 3)
# print(y_train.shape)        #(23091,)
# print(np.unique(y_train,return_counts=True))    #(array([0., 1.], dtype=float32), array([15001,  8090], dtype=int64))
num_data_men = np.unique(y_train,return_counts=True)[1][0]          #15001
num_data_women = np.unique(y_train,return_counts=True)[1][1]        #8090
# exit()    
woman_idx = np.where(y_train==1)
# print(woman_idx)                #(array([    1,     3,     8, ..., 23087, 23089, 23090], dtype=int64),)
women_data =[]
for idx in woman_idx:
    women_data.append(x_train[idx])

women_x_data = np.array(women_data).reshape(-1,100,100,3)
women_y_data = np.ones(shape=women_x_data.shape[0])
# print(women_x_data.shape)             #(8090, 100, 100, 3)

datagen = ImageDataGenerator(
    # rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    # zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)

augment_size = num_data_men-num_data_women
# print(augment_size)
# exit()

randidx = np.random.choice(women_x_data.shape[0], size=augment_size)
# print(randidx)                  #[1093 6010 1632 ... 1495 2952 4427]
# print(randidx.shape)            #(6911,)
# print(len(randidx))             #6911

x_augmented = women_x_data[randidx].copy()
y_augmented = women_y_data[randidx].copy()

# print(x_augmented.shape,y_augmented.shape)          #(6911, 100, 100, 3) (6911,)

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2],3)

x_augmented = datagen.flow(
    x_augmented, y_augmented,
    batch_size = augment_size,
    shuffle = False,
).next()[0]

# print(x_augmented.shape)       # (6911, 100, 100, 3)

x_train =  np.concatenate((x_train,x_augmented))
y_train =  np.concatenate((y_train,y_augmented))
# print(x_train.shape, y_train.shape)                 # (30002, 100, 100, 3) (30002,)
# exit()

########################### 스케일링 1. ###########################
# x_train = x_train/255.
# x_test = x_test/255.

# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(sparse_output=False)
# y_train = ohe.fit_transform(y_train.reshape(-1,1))
# y_test = ohe.fit_transform(y_test.reshape(-1,1))

#2. 모델 구성
model = Sequential()
model.add(Conv2D(32, (20,20), padding='same', input_shape = (100,100,3)))                      #(26,26,64)
model.add(MaxPooling2D())
model.add(Conv2D(filters=16 , kernel_size=(10,10), padding='same', activation='relu'))        #(24,24,32)
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Conv2D(16,(5,5), padding='same',activation='relu'))                               #(23,23,32)
model.add(MaxPooling2D())
model.add(Conv2D(8,(5,5), padding='same',activation='relu'))                               #(22,22,16)
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Conv2D(4,(3,3), padding='same',activation='relu'))                               #(21,21,16)

model.add(GlobalAveragePooling2D())                                                        #이후 FC layer와 붙기 위해 한줄로 reshape

model.add(Dense(16, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=4, activation='relu'))
model.add(Dense(1,activation='sigmoid'))    

model.summary()
# exit()

#3. 컴파일, 훈련
import time
from keras.optimizers import Adam
model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate=0.01),
              metrics = ['acc'])
start_time = time.time()
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=50,
                   restore_best_weights=True)
from keras.callbacks import ReduceLROnPlateau
rlr = ReduceLROnPlateau(monitor='val_loss',
                  mode='auto',
                  patience=20,
                  verbose=1,
                  factor=0.5,
                  )
model.fit(x_train,y_train,
          epochs = 2000, batch_size = 500,
          verbose = 1,
          validation_split = 0.3,
          callbacks = [es,rlr])
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
from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test,np.round(y_predict))
print('accuraccy_score: ',acc_score)
print('소요시간 : ',round(end_time-start_time,2), '초')


# accuraccy_score:  0.7085377821393523
# 소요시간 :  1085.39 초

# accuraccy_score:  0.7769872423945045
# 소요시간 :  2438.09 초