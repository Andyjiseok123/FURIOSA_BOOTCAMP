# https://www.kaggle.com/competitions/bike-sharing-demand/data
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

import datetime
path = './_save/keras31/kaggle_bike/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:4f}.keras'
filepath = "".join([path,"k31_",date,filename])

#1. 데이터
path = './_data/bike-sharing-demand/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# print(train_csv)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# print(test_csv)
submission = pd.read_csv(path +'sampleSubmission.csv',index_col=0)

################# x,y 분리 ##########################
x = train_csv.drop(['casual','registered', 'count'], axis=1)
# print(x) #[10886 rows x 8 columns]

y = train_csv['count']
# print(y, y.shape)  ##(10886,)

x_train,x_test, y_train, y_test = train_test_split(x,y,
                 train_size=0.8,
                 random_state=999,

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
# model.add(Dense(10, activation='relu', input_dim=8))
# model.add(Dropout(0.2))
# model.add(Dense(20,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(30,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(20,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(10,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1,activation='relu'))

input1 = Input(shape=(8,))
dense1 = Dense(10,activation = 'relu',name='ys1')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(20,activation = 'relu', name= 'ys2')(drop1)
drop2 = Dropout(0.2)(dense2)
dense3 = Dense(30,activation = 'relu', name= 'ys3')(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(20,activation = 'relu', name= 'ys4')(drop3)
drop4 = Dropout(0.2)(dense4)
dense5 = Dense(10,activation = 'relu', name= 'ys5')(drop4)
drop5 = Dropout(0.2)(dense5)
output1 = Dense(1)(drop5)

model = Model(inputs=input1, outputs=output1)

#3.컴파일 훈련
model.compile(loss = 'mse', optimizer= 'adam' )
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(
    monitor= 'val_loss',
    mode='auto',
    patience=15,
    restore_best_weights=True
)
mcp = ModelCheckpoint(monitor='val_loss',
                      mode='auto',
                      save_best_only=True,
                      filepath = filepath,
                      verbose=1
                      )

hist = model.fit(x_train,y_train, 
                 epochs = 1000, batch_size=200, validation_split=0.33,
                 callbacks = [es,mcp])

#4.평가 ,예측
loss = model.evaluate(x_test,y_test)
print("loss:", loss)

y_predict = model.predict(x_test)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict) 
print('r2결과값: ' ,r2)

mse = mean_squared_error(y_test,y_predict)
print('mse : ', mse)

def RMSE(y_test, y_predict):  #RMSE 함수정의
    return np.sqrt(mean_squared_error(y_test,y_predict))  #np.sqrt하면 mse에 루트가 씌워짐

rmse = RMSE(y_test, y_predict)

print('RMSE : ', rmse) 


# r2결과값:  0.32226014137268066
# mse :  21475.21875
# RMSE :  146.54425526099615