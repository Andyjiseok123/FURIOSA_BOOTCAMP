# acc : 1.0

import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. 데이터
datasets = load_digits()
# print(datasets.data.shape)              # (1797, 64)
# print(datasets.feature_names)           
# exit()
x = datasets.data
y = datasets.target
# print(x.shape)    # (1797, 64)
# print(y.shape)    # (1797,)
# print(y)

# print(np.unique(y,return_counts=True))          #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180]))
# exit()


##################### 원핫 1. to_categorical ######################
#####################    from tensorflow   #######################
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# to_categorical을 사용하는 경우 target의 데이터셋이 0부터 시작하지 않을때 강제로 0부터 시작하도록 함.
# print(y.shape)
# exit()

##################### 원핫 2. pd.get_dummies ######################
#######################    from pandas   #########################
y = pd.get_dummies(y,dtype=int)
# print(y)


####################### 원핫 3. sklearn.processing ######################
#######################    from sci-kit learn   #########################
# from sklearn.preprocessing import OneHotEncoder
# category = y.reshape(-1,1)
# print(category)
# encoder = OneHotEncoder().fit_transform(category)
# y = encoder.toarray()
# print(y)


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.75,
                                                    shuffle= True,
                                                    random_state=333,
                                                    stratify=y)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# exit()


#2. 모델 구성
model = Sequential()
model.add(Dense(54,activation='relu',input_dim=64))
model.add(Dense(150,activation='relu'))
model.add(Dense(300,activation='relu'))
model.add(Dense(900,activation='relu'))
model.add(Dense(1300,activation='relu'))
model.add(Dense(900,activation='relu'))
model.add(Dense(450,activation='relu'))
model.add(Dense(300,activation='relu'))
model.add(Dense(10,activation='softmax'))

#3. 컴파일 푼련
model.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=100,
    restore_best_weights=True
    )

start_time = time.time()
model.fit(x_train, y_train,epochs = 3000, batch_size = 10,
          verbose=1,
          validation_split = 30/75,
          callbacks = [es]
          )
end_time = time.time()


#4. 평가 예측
result = model.evaluate(x_test,y_test)
print("loss = ", result[0])
print("acc = ", round(result[1],2))

y_predict = model.predict(x_test)

# print(y_predict)
# print(y_predict[0])
# print(np.argmax(y_predict[0]))
y_predict = np.argmax(y_predict,axis=1)
y_test = np.argmax(y_test, axis=1)
# print(y_predict)
# print(y_test)

accuracy_score = accuracy_score(y_test,y_predict)
print("acc_score : ", accuracy_score)
print("소요시간 : ", round(end_time-start_time,2),"초")