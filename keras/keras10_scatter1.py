import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,7,5,7,8,6,10])

# [검색] train과 test를 섞어서 7:3 나눈다
# 힌트 : 사이킷런
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size = 0.7,   #default = 0.75
                                                    shuffle=True,       #default = True
                                                    random_state = 312)

print("x_train : ", x_train)
print("y_train : ", y_train)
print("x_test : ", x_test)
print("x_test : ", y_test)

#2. 모델 구성
model = Sequential()
model.add(Dense(3,input_dim = 1))
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 2000, batch_size = 3)

print("=========================================================")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x)

print("loss : ", loss)
# print("11의 예측값 : ", results)

# 그래프 그리기
import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.plot(x, results, color='red')
plt.show()
