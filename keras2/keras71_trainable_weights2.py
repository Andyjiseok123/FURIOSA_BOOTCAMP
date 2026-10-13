import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

print(tf.__version__)           #2.21.0

# 가중치 초기값의 random값을 고정하고 싶을때 사용
random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

#1. 데이터
x = np.array([1,2,3,4,5,])
y = np.array([1,2,3,4,5,])

#2. 모델
model = Sequential()
model.add(Dense(3,input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

###################################### 동결 ######################################
model.trainable = False
# model.trainable = True
###################################### 동결 ######################################

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=1000,batch_size=1,verbose=1)

#4. 평가, 예측
y_predict = model.predict(x)
print(y_predict)

# model.trainable = False인 경우 weight가 동결되어 loss가 줄지 않음