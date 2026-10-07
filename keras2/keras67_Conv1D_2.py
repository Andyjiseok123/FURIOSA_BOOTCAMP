# 55-2 copy

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

#1. 데이터
x=[1,2,3,4,5,6,7,8,9,10,11,12,20,30,40,50,60,70]
x=np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6,],
            [5,6,7],[6,7,8],[7,8,9],[8,9,10],
            [9,10,11],[10,11,12],
            [20,30,40],[30,40,50],[40,50,60]])
y=np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])        #80에 맞춰보아요

x=x.reshape(13,3,1)
x_predict = x_predict.reshape(1,3,1)
#2. 모델 구성
model = Sequential()
model.add(SimpleRNN(10,input_shape = (3,1)))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
lr=0.01
model.compile(loss='mse', optimizer=Adam(learning_rate=lr))
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   verbose=1,
                   patience=40,
                   restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_loss',
                        mode='min',
                        verbose=1,
                        patience=20,
                        factor=0.5,
                        )
model.fit(x,y,epochs=2000,verbose=1,validation_split=0.1,
          callbacks=[es,rlr])

#4. 평가, 예측
loss = model.evaluate(x,y)
y_predict = model.predict(x_predict)
print("loss: ",loss)
print("y_predict:", y_predict )
