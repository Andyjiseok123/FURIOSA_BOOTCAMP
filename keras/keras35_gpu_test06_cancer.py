import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.datasets import load_breast_cancer #유방암 관련 데이터셋 불러오기
from sklearn.metrics import r2_score,mean_squared_error
import os
import datetime
path = './_save/keras31/cancer/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:4f}.keras'
filepath = "".join([path,"k31_",date,filename])


#1.데이터

datasets = load_breast_cancer()


x = datasets.data #(569, 30)
y = datasets.target #(569,)

x_train,x_test,y_train,y_test =train_test_split(
    x,y,
    random_state=333,
    train_size=0.8,
    stratify=y,  #x,y데이터를 나눌떄 stratify=y이걸안넣으면 x,y서로 데이터 크기가 달랐을떄 비율편차가 생길수있음
)
from sklearn.preprocessing import MinMaxScaler,StandardScaler ,MaxAbsScaler
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

# print(np.unique(y_train,return_counts=True))
# # (array([0, 1]), array([175, 280])) #>>>startify 적용후 (array([0, 1]), array([170, 285]))
# print(np.unique(y_test,return_counts=True))
# # (array([0, 1]), array([37, 77]))  #>>>startify 적용후 (array([0, 1]), array([42, 72]))

# print(x_train.shape,x_test.shape)  #(455, 30) (114, 30)
# print(y_train.shape,y_test.shape)  #(455,) (114,)



#2모델구성
# model = Sequential()
# model.add(Dense(30, input_dim=30, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(60, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(70, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(80, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(60, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))  #기본 디폴트값은 리니어 
# model.add(Dense(1, activation='sigmoid'))  #마지막은 무조건 시그모이드 고정  

input1 = Input(shape=(30,))
dense1 = Dense(30,activation = 'relu',name='ys1')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(60,activation = 'relu', name= 'ys2')(drop1)
drop2 = Dropout(0.2)(dense2)
dense3 = Dense(70,activation = 'relu', name= 'ys3')(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(80,activation = 'relu', name= 'ys4')(drop3)
drop4 = Dropout(0.2)(dense4)
dense5 = Dense(60,activation = 'relu', name= 'ys5')(drop4)
drop5 = Dropout(0.2)(dense5)
dense6 = Dense(60,activation = 'relu', name= 'ys6')(drop5)
output1 = Dense(1,activation='sigmoid')(dense6)

model = Model(inputs=input1, outputs=output1)


#3.컴파일 ,훈련
model.compile(loss ='binary_crossentropy',
                optimizer= 'adam',
                metrics=['acc'],        
            )  #이진분류에서는 loss = 'binary_crossetropy' 고정
es = EarlyStopping(
            monitor='val_loss',
            mode= 'auto',
            patience=20,
            restore_best_weights=True,
            )
mcp = ModelCheckpoint(monitor='val_loss',
                      mode='auto',
                      save_best_only=True,
                      filepath = filepath,
                      verbose=1
                      )
strat_time = time.time()  #현재 시간을 반환 ,시작시간
model.fit(x_train,y_train ,
           epochs= 500 , 
           batch_size=32,
           validation_split=0.2,
           callbacks =[es,mcp], 
           )
end_time = time.time()  #현재 시간을 반환 ,시작시간

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('==============================')
print("loss:", loss[0])
print('acc:',round(loss[1],4)) #loss: 0번[0.12450382113456726,###LOSS값 (1번) 0.9473684430122375]###ACC값
print('==============================')
y_pred = model.predict(x_test)  #시그모이드 함수를 거쳐 0,1사이 값을 반환후 >>metrics=['acc']로 후처리하면 0 OR 1로 반올림내림해서 퍼센테이지로 변환
y_pred = np.round(y_pred)# y_pred한 값이 0.11121515,0.125148이런식으로 나와서 라운드처리후  [1.] 이런식으로 변환한다음에 acc값 비교 이거 안하면 에러남
from sklearn.metrics import accuracy_score

acc_score = accuracy_score(y_test,y_pred,normalize=True)
print('acc_score:', acc_score)  #acc_score: 0.9298245614035088


# loss: 0.13742943108081818
# acc: 0.9561
# ==============================
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
# acc_score: 0.956140350877193