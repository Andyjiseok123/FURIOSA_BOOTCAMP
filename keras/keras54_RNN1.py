import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9]])

y= np.array([4,5,6,7,8,9,10])

# print(x.shape, y.shape)     #(7, 3) (7,)

x = x.reshape(x.shape[0],
              x.shape[1],
              1)

#2. 모델 구성
model = Sequential()
model.add(SimpleRNN(units=10, input_shape=(3,1)))       # 3차원으로 들어가서 2(1)차원으로 나옴 -> 바로 Dense와 연결가능
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
lr = 0.05
es = EarlyStopping(monitor='loss',
                   mode='auto',
                   patience=400,
                   verbose=1,
                   restore_best_weights=True
                   )
rlr = ReduceLROnPlateau(monitor='loss',
                        mode='auto',
                        patience=20,
                        verbose=1,
                        factor=0.1
                        )
model.compile(loss='mse',optimizer=Adam(learning_rate=lr))
model.fit(x,y,epochs=20000,callbacks=[es,rlr])

#4. 평가, 예측
result = model.evaluate(x,y)
print('loss:', result)

x_predict = np.array([8,9,10]).reshape(1,3,1)
y_predict = model.predict(x_predict)

print('[8,9,10]의 결과:', y_predict)

# [8,9,10]의 결과: [[10.678212]]

# [8,9,10]의 결과: [[11.113018]]