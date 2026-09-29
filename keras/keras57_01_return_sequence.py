import numpy as np
from keras.models import Sequential
from keras.layers import Dense,LSTM,SimpleRNN,GRU, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from my_lib import split_x

#1. 데이터 55-2 copy
x=[1,2,3,4,5,6,7,8,9,10,11,12,20,30,40,50,60,70]
x=np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6,],
            [5,6,7],[6,7,8],[7,8,9],[8,9,10],
            [9,10,11],[10,11,12],
            [20,30,40],[30,40,50],[40,50,60]])
y=np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x = x.reshape(-1,3,1)
x_predict = np.array([50,60,70])        #80에 맞춰보아요
x_predict = x_predict.reshape(-1,3,1)
#2. 모델 구성
model = Sequential()
model.add(LSTM(units=10, input_shape=(3,1),return_sequences=True))          #return_sequences=true : 은닉층의 값이 다음 LSTM같은 layer로 넘어가짐
model.add(LSTM(5,return_sequences=True))
model.add(LSTM(5))
model.add(Dense(8))
model.add(Dense(1))
# model.summary()

#3. 컴파일, 훈련
lr=0.001
model.compile(loss='mse', optimizer=Adam(learning_rate=lr))
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   verbose=1,
                   patience=200,
                   restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_loss',
                        mode='min',
                        verbose=1,
                        patience=100,
                        factor=0.5,
                        )
model.fit(x,y,epochs=30000,verbose=1,validation_split=0.1,
          callbacks=[es,rlr]
          )

#4. 평가, 예측
loss = model.evaluate(x,y)
y_predict = model.predict(x_predict)
# y_predict=[]
# for i in range(len(x_predict)):
#     y_predict.append(model.predict(x_predict[i]))

print("loss: ",loss)
print("y_predict:", y_predict )