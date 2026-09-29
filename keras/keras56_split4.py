import numpy as np
from my_lib import split_x
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

a = np.array(range(1,101))
x_predict = np.array(range(96,106))
size = 6

a = a.reshape(-1,2)
# print(a.shape)

#1. 데이터
xy_data = split_x(a,size)

# print(xy_data.shape)            #(45, 6, 2)
x = xy_data[:,:-1].reshape(-1,5,2)
y = xy_data[:,-1,-1]

# print(x.shape)                  #(45, 5, 2)
# print(y.shape)                  #(45,)
# exit()
x_predict = split_x(x_predict,size-1).reshape(-1,5,2)
# print(x_predict)
# print(x_predict.shape)            #(3, 5, 2)
# exit()

#2. 모델 구성
model = Sequential()
model.add(LSTM(30,input_shape = (5,2)))
model.add(Dense(15,activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(6,activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(3,activation='tanh'))
model.add(Dense(1))

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
model.fit(x,y,epochs=30000,verbose=1,validation_split=0.3,
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