from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

# [실습] train_test_split으로 자르기
x_train_val , x_test, y_train_val , y_test = train_test_split(x,y,train_size=12/len(x),random_state=312)
x_train , x_val, y_train , y_val = train_test_split(x_train_val,y_train_val,train_size=8/len(x_train_val),random_state=444)

# print(x_train)
# print(x_val)
# print(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(4,input_dim=1))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 10, validation_data = (x_val,y_val))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print(loss)