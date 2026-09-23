import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM

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
# model.add(SimpleRNN(units=10, input_shape=(3,1)))       # 3차원으로 들어가서 2(1)차원으로 나옴 -> 바로 Dense와 연결가능
model.add(LSTM(units=3, input_shape=(3,1)))                
model.add(Dense(7,activation='relu'))
model.add(Dense(1))

model.summary()
# 파라미터의 갯수 = units*(feature + bias + units)
exit()

"""LSTM
 Layer (type)                Output Shape              Param #   
=================================================================
 simple_rnn (SimpleRNN)      (None, 10)                480       
 dense (Dense)               (None, 7)                 77        
 dense_1 (Dense)             (None, 1)                 8         
=================================================================
Total params: 205
Trainable params: 205
Non-trainable params: 0

_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 1)                 12        
                                                                 
 dense (Dense)               (None, 7)                 14        
                                                                 
 dense_1 (Dense)             (None, 1)                 8         
                                                                 
=================================================================
Total params: 34
Trainable params: 34
Non-trainable params: 0

_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 2)                 32        
                                                                 
 dense (Dense)               (None, 7)                 21        
                                                                 
 dense_1 (Dense)             (None, 1)                 8         
                                                                 
=================================================================
Total params: 61
Trainable params: 61
Non-trainable params: 0

"""




"""
가중치의 누적때문에 RNN은 초기 time step때의 데이터가 손실된다
이를 해결하기 위해 나온 LSTM
"""