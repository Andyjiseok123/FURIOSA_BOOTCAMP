import warnings
warnings.filterwarnings("ignore")

from keras.applications import VGG16, VGG19
from keras.applications import ResNet50, ResNet50V2
from keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from keras.applications import DenseNet201, DenseNet169, DenseNet121
from keras.applications import InceptionV3, InceptionResNetV2
from keras.applications import MobileNet, MobileNetV2
from keras.applications import MobileNetV3Small, MobileNetV3Large
from keras.applications import NASNetMobile, NASNetLarge
from keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2
from keras.applications import Xception

model_list = [VGG19(include_top=False, input_shape=(32,32,3)),
              ResNet50(include_top=False, input_shape=(32,32,3)),
              ResNet101(include_top=False, input_shape=(32,32,3)),
              DenseNet201(include_top=False, input_shape=(32,32,3)),
            #   InceptionV3(), 
            #   InceptionResNetV2(),
              MobileNetV2(include_top=False, input_shape=(32,32,3)),
              EfficientNetB0(include_top=False, input_shape=(32,32,3))
              ]

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
import tensorflow as tf
import time
from sklearn.metrics import accuracy_score
import random

random.seed(312)
np.random.seed(312)
tf.random.set_seed(312)

from keras.datasets import cifar10, cifar100

for model_i in model_list:
    model_i.trainable = False

    model = Sequential()
    model.add(model_i)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(32,activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(10,activation='softmax'))

    (x_train, y_train), (x_test,y_test) = cifar10.load_data()

    # 스케일링
    x_train = x_train/255.
    x_test = x_test/255.

    #3. 컴파일 훈련
    model.compile(loss='sparse_categorical_crossentrpy', optimizer='adam', metrics=['acc'])     #onehot encoding 안해줘서 loss를 저거 씀
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    es = EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor='val_loss', mode = 'min', patience=50, factor=0.5)
    model.fit(x_train,y_train,epochs=100000, batch_size=32,
              verbose=0, validation_split=0.2,
              callbacks=[es,rlr])

    #4. 평가, 예측
    loss = model.evaluate(x_test,y_test, verbose=0)
    print("=========================================================")
    print("모델명 : ",model_i.name, "loss : ",loss[0], "acc : ",loss[1])