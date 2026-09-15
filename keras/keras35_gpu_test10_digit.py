from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Input
import time
from sklearn.metrics import accuracy_score
import os
import datetime
path = './_save/keras31/digits/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:4f}.keras'
filepath = "".join([path,"k31_",date,filename])

#1데이터

datasets = load_digits()
# print(datasets)


x = datasets.data
y = datasets.target
# print(x.shape,y.shape) #(1797, 64) (1797,)

y = pd.get_dummies(y,dtype=int)
# print(y)

# print(np.unique(y))  #[0 1]
# print(np.unique(y,return_counts=True)) #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180]))
# print(pd.Series(y).value_counts())

x_train,x_test,y_train,y_test =train_test_split(
    x,y,
    random_state=333,
    train_size=0.8,
    stratify=y,  #x,y데이터를 나눌떄 stratify=y이걸안넣으면 x,y서로 데이터 크기가 달랐을떄 비율편차가 생길수있음
)
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler 
from sklearn.preprocessing import RobustScaler
##############################################################################
# scaler = MinMaxScaler()
##############################################################################



##############################################################################
# scaler = StandardScaler()
##############################################################################



##############################################################################
# scaler = MaxAbsScaler()
##############################################################################


##############################################################################
scaler = RobustScaler()
##############################################################################
scaler.fit(x_train) # x 값을  MinMaxScaler으로 실행시킬 준비
x_train = scaler.fit_transform(x_train) # 0~1 값 변환 사이로변환
x_test = scaler.transform(x_test) 

#2.모델구성
# model = Sequential()
# model.add(Dense(30, input_dim=64, activation= 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(50, actzvation= 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(100,activation= 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(50, activation= 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(30, activation= 'relu'))
# model.add(Dense(10,activation='softmax'))

input1 = Input(shape=(64,))
dense1 = Dense(30,activation = 'relu',name='ys1')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(50,activation = 'relu', name= 'ys2')(drop1)
drop2 = Dropout(0.2)(dense2)
dense3 = Dense(100,activation = 'relu', name= 'ys3')(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(50,activation = 'relu', name= 'ys4')(drop3)
drop4 = Dropout(0.2)(dense4)
dense5 = Dense(30,activation = 'relu', name= 'ys5')(drop4)
output1 = Dense(10,activation='softmax')(dense5)

model = Model(inputs=input1, outputs=output1)

#3.컴파일,훈련
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics =['acc']
              )
start_time =time.time()
model.fit(x_train,y_train, epochs=500,batch_size=300,
          verbose=1,
          validation_split=0.3,
          )
end_time =time.time()

result = model.evaluate(x_test,y_test,)
print('loss: ',result[0])
print('acc: ',round(result[1],2))
y_predict= model.predict(x_test) 

y_predict = np.argmax(y_predict,axis=1) 
# print(y_predict)#[0 2 0 1 1 2 0 2 0 2 2 1 2 0 0 0 2 0 2 1 0 2 1 1 0 2 1 1 1 2]
y_test = np.argmax(y_test, axis=1)
# print(y_test) #[0 2 0 1 1 1 0 2 0 2 2 2 2 0 0 0 2 0 2 1 0 2 1 1 0 2 1 1 1 1]
# #######################################################
# y_predict = np.argmax(model.predict(x_test),axis =1)
# y_test_argmax =np.argmax(y_test,axis=1)
# ########################################################
# y_predict = model.predict(x_test)
# print(y_predict)


accuracy_score =accuracy_score(y_test,y_predict)  
#지금까지는 y_predict 값은 [0.7,0.2,0.1]이런식으로 되어있어서 비교가 불가능함 >>가장큰 수를 1로 바꿔줘야함 그래서 결과를 [1,0,0]으로 변경후 비교 
print('acc_score :',accuracy_score)
print('걸린시간: ', round(end_time-start_time, 2),'초')

# CPU
# acc_score : 0.9638888888888889
# 걸린시간:  30.69 초

# GPU
# acc_score : 0.9694444444444444
# 걸린시간:  15.82 초