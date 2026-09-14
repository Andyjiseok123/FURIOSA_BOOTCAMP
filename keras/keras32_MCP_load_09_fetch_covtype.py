from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense 
import time
from sklearn.metrics import accuracy_score
import os

import datetime
path = './_save/keras31/fetch_covtype/'
filename = os.listdir(path)[-1]
filepath = "".join([path,filename])


#1.데이터
datasets = fetch_covtype()
# print(datasets)
# shape=(581012, 54))  ,shape=(581012,)

x = datasets.data
y = datasets.target

# print(x.shape,y.shape)  #(581012, 54) (581012,)
print(np.unique(y,return_counts=True)) 

# from tensorflow.keras.utils import to_categorical 
# y = to_categorical(y)
# print(y)
# print(y.shape) #(581012, 8)
'''
#(array([1, 2, 3, 4, 5, 6, 7], dtype=int32),
#array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))
>>>>.
to_categorical  쓰면 0~부터 컬럼을 만들어서 만약 1,2,3,4,5,6,7의 컬럼이 형성되어있으면 
0,1,2,3,4,5,6,7,8로 늘어남 
'''
y = pd.get_dummies(y,dtype=int)
# print(y.shape) #(581012, 7)


x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    train_size=0.7,
    random_state=333,
    shuffle=True,
    stratify=y,
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
model = load_model(filepath)

#3.컴파일,훈련


result = model.evaluate(x_test,y_test,)
print('loss: ',result[0])
print('acc: ',round(result[1],2))
y_predict= model.predict(x_test) 

y_predict = np.argmax(y_predict,axis=1) 
print(y_predict)#[0 2 0 1 1 2 0 2 0 2 2 1 2 0 0 0 2 0 2 1 0 2 1 1 0 2 1 1 1 2]
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

# loss:  0.16308407485485077
# acc:  0.94
# 5447/5447 ━━━━━━━━━━━━━━━━━━━━ 4s 647us/step 
# [1 1 5 ... 1 1 0]
# acc_score : 0.9412807508720397