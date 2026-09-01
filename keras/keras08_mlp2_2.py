import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
# x = np.array(range(10))             # x = [0 1 2 3 4 5 6 7 8 9]

# x = np.array(range(1,10))           # x = [1 2 3 4 5 6 7 8 9]

x = np.array([range(10),range(21,31),range(201,211)]).T #(3,10) -> (10,3)

y = np.array(range(1,11))

#2. 모델 구성
model = Sequential()
model.add(Dense(5,input_dim=3))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2000, batch_size=4)

#4. 평가, 예측
loss = model.evaluate(x,y)
results = model.predict(np.array([[10,31,211]]))
print("loss : ", loss)
print("결과값 : ", results)

#[10,31,211] 예측 11.00 합격