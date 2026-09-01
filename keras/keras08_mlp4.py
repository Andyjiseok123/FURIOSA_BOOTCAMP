import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(range(10))
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [10,9,8,7,6,5,4,3,2,1],
              [9,8,7,6,5,4,3,2,1,0]]).T


#2. 모델 구성
model = Sequential()
model.add(Dense(5,input_dim=1))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2000, batch_size=4)

#4. 평가, 예측
loss = model.evaluate(x,y)
results = model.predict(np.array([[10]]))
print("loss : ", loss)
print("결과값 : ", results)

