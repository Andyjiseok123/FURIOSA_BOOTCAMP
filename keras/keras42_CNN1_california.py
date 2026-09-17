#29-6 카피

# import ssl
# ssl._create_default_https_context = ssl.create_default_context 다운로드 안될떄 사용할것
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import time

# path = './_save/keras30/'

#1.데이터 
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# print(x.shape)    #(20640, 8)
x = x.reshape(-1,4,2,1)
# print(x.shape)    #(20640, 4, 2, 1)
# print(y.shape)
# exit()


x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    train_size=0.75,
    random_state=333
)

from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler  #preprocessing(전처리)
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
# scaler = RobustScaler()
##############################################################################
# 이상치에 강력함

##############################################################################
# x_train = scaler.fit_transform(x_train)
##############################################################################
# x_test = scaler.transform(x_test) 

#2.모델구성
model = Sequential()
model.add(Conv2D(64, (2,2),padding='same', input_shape = (4,2,1)))                      #(26,26,64)
model.add(Dropout(0.3))
model.add(Conv2D(32,(2,2),padding='same',activation='relu'))                               #(23,23,32)
model.add(Dropout(0.2))
model.add(Conv2D(16,(3,3),padding='same',activation='relu'))                               #(20,20,16)

# model.add(Flatten())                                                        #이후 FC layer와 붙기 위해 한줄로 reshape
model.add(GlobalAveragePooling2D())

model.add(Dense(128, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(1)) 

# model.summary()
# exit()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#3.컴파일,훈련
model.compile(loss='mse', optimizer= 'adam')

es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=30,
                   restore_best_weights=True,
                   verbose=1
                   )

# mcp = ModelCheckpoint(monitor='val_loss',
#                       mode='auto',
#                       save_best_only=True,
#                       filepath=path + 'keras30_mcp1.keras',
#                       verbose=1
#                       )

start_time = time.time()  #현재 시간을 반환 ,시작시간
hist = model.fit(x_train,y_train, 
                 epochs=1000, 
                 batch_size=64,
                 validation_split=0.2,
                 verbose = 1)
end_time = time.time()  #훈련 끝난 시간을 반환 , 끝시간

#######################################################
# model.save(path + 'keras29_3_save_model.keras') #가중치 세이브
#######################################################

#4.평가 ,예측
loss = model.evaluate(x_test,y_test)
print("loss:", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict) 
print('r2결과값: ' ,r2)

mse = mean_squared_error(y_test,y_predict)
print('mse : ', mse)

def RMSE(y_test, y_predict):  #RMSE 함수정의
    return np.sqrt(mean_squared_error(y_test,y_predict))  #np.sqrt하면 mse에 루트가 씌워짐

rmse = RMSE(y_test, y_predict)

print('RMSE : ', rmse) 
print('걸린시간: ', round(end_time-start_time, 2),'초')

# CPU
# r2결과값:  0.6934426989119218
# mse :  0.39228786385406883
# RMSE :  0.6263288783491217
# 걸린시간:  239.8 초

#GPUr2
# 결과값:  0.6637992773327828
# mse :  0.43022124364091946
# RMSE :  0.6559125274309978
# 걸린 시간:  434.51 초