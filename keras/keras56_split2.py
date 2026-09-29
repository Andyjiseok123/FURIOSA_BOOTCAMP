import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

a = np.array([[1,2,3,4,5,6,7,8,9,10],
             [9,8,7,6,5,4,3,2,1,0],]).T
size=3

def split_x(dataset, size):
    x = []
    y = []
    for i in range(len(dataset)-size):
        subset = dataset[i:(i+size+1)]
        x.append(subset[:size])
        y.append(subset[size:][-1][-1])
    return np.array(x), np.array(y)

x,y = split_x(a,size)

# print(x.shape)
# print(y.shape)
# print(x)
# print(y)

#2. 모델 구성
model = Sequential()
model.add(LSTM(10,input_shape = (3,2)))
model.add(Dense(12))
model.add(Dropout(0.2))
model.add(Dense(12))
model.add(Dropout(0.2))
model.add(Dense(12))
model.add(Dropout(0.2))
model.add(Dense(12))
model.add(Dropout(0.2))
model.add(Dense(12))
model.add(Dropout(0.2))
model.add(Dense(6))
model.add(Dropout(0.2))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
lr=0.01
model.compile(loss='mse', optimizer=Adam(learning_rate=lr))
# es = EarlyStopping(monitor='val_loss',
#                    mode='min',
#                    verbose=1,
#                    patience=40,
#                    restore_best_weights=True)
# rlr = ReduceLROnPlateau(monitor='val_loss',
#                         mode='min',
#                         verbose=1,
#                         patience=20,
#                         factor=0.5,
#                         )
model.fit(x,y,epochs=2000,verbose=1,validation_split=0.1,
        #   callbacks=[es,rlr]
          )

#4. 평가, 예측
x_predict=np.array([[8,2],[9,1],[10,0]]).reshape(1,3,2)
loss = model.evaluate(x,y)
y_predict = model.predict(x_predict)
print("loss: ",loss)
print("y_predict:", y_predict )