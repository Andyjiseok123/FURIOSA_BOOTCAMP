import os
os.environ["TF_GOU_ALLOCATOR"] = "cuda_malloc_async"        #메모리 터지는거 어느정도 방지

import numpy as np
import pandas as pd
from my_lib import split_x
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, SimpleRNN
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error


path = './_data/jena_climate/'
data_csv = 'jena_climate_2009_2016.csv'
submission_csv = 'jena_climate_submission.csv'
df = pd.read_csv(path + data_csv,index_col=0)
# print(type(df))                 #<class 'pandas.core.frame.DataFrame'>
# print(df.shape)                 #(420551, 14)

x = df.drop('T (degC)',axis=1)
y = df['T (degC)']


# print(x.shape)                  #(420551, 13)
# print(y.shape)                  #(420551,)

x = x.to_numpy()
y = y.to_numpy()

# print(type(x))                  #<class 'numpy.ndarray'>
# print(type(y))                  #<class 'numpy.ndarray'>
# print(x.shape)                  #(420551, 13)
# print(y.shape)                  #(420551,)

################## 자르기 전 x scaling ########################
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
##############################################################
x_train = x[:-288]
y_train = y[144:-144]
x_test = x[-288:-144]
y_test = y[-144:]
# print(x_train.shape)            #(420263, 13)
# print(y_train.shape)            #(420263,)
# print(x_test.shape)             #(144, 13)
# print(y_test.shape)             #(144,)

time_step = 144
x_train = split_x(x_train,time_step)
y_train = split_x(y_train,time_step)

# print(x_train.shape)            #(420120, 144, 13)
# print(y_train.shape)            #(420120, 144)

#2. 모델 구성
model = Sequential()
model.add(Bidirectional(SimpleRNN(units=200), input_shape=(144,13)))          #return_sequences=true : 은닉층의 값이 다음 LSTM같은 layer로 넘어가짐
model.add(Dense(72))
model.add(Dense(144))
# model.summary()

#3. 컴파일, 훈련
lr=0.05
model.compile(loss='mse', optimizer=Adam(learning_rate=lr))
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   verbose=1,
                   patience=200,
                   restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_loss',
                        mode='min',
                        verbose=1,
                        patience=50,
                        factor=0.5,
                        )
model.fit(x_train,y_train,
          epochs=30000, batch_size=1000,
          verbose=1,validation_split=0.3,
          callbacks=[es,rlr]
          )

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
mse = mean_squared_error(y_test,y_predict)
rmse = np.sqrt(mse)

print("loss: ",loss)
print('r2결과값: ' ,r2)
print('mse : ', mse)
print('RMSE : ', rmse) 

y_predict.to_csv(path+submission_csv)