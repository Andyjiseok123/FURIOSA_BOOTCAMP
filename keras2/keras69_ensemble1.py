import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input
import time
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

#1. 데이터
x1_datasets = np.array([range(100),range(301,401)]).T
x2_datasets = np.array([range(101,201),range(411,511),range(150,250)]).transpose()

y = np.array(range(3001,3101))

x1_train,x1_test,x2_train,x2_test,y_train,y_test = train_test_split(x1_datasets,x2_datasets,y,train_size=0.85,random_state=414)
# 순서확인할것 
# print(x1_train.shape)           # (85,2)
# print(x2_train.shape)           # (85,3)
# print(x1_test.shape)            # (15,2)
# print(x2_test.shape)            # (15,3)
# print(y_train.shape)            # (85,)
# print(y_test.shape)             # (15,)
# exit()

#2-1 모델
input1 = Input(shape=(2,))
dense1 = Dense(10,activation='relu',name='han1')(input1)
dense2 = Dense(20,activation='relu',name='han2')(dense1)
dense3 = Dense(30,activation='relu',name='han3')(dense2)
output1 = Dense(5,activation='relu',name='han4')(dense3)
# model1 = Model(inputs=input1, outputs=output1)            #merge할때 필요가 없음

#2-2 모델
input21 = Input(shape=(3,))
dense21 = Dense(50,name='han21')(input21)
dense22 = Dense(40,name='han22')(dense21)
dense23 = Dense(30,name='han23')(dense22)
dense24 = Dense(20,name='han24')(dense23)
dense25 = Dense(10,name='han25')(dense24)
output21 = Dense(3,name='han26')(dense25)
# model21 = Model(inputs=input21, outputs=output21)         #merge할때 필요가 없음

#2-3 모델 합치기
from keras.layers import concatenate, Concatenate

merge1 = concatenate([output1, output21],name='mg1')
# merge1 = Concatenate(neme='mg1')([output1, output21])
merge2 = Dense(10, name='mg2')(merge1)
merge3 = Dense(5, name='mg3')(merge2)
last_output = Dense(1,name='last')(merge3)

model = Model(inputs=[input1,input21],outputs=last_output)

model.summary()

"""
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ input_layer_1 (InputLayer)    │ (None, 3)                 │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ han21 (Dense)                 │ (None, 50)                │             200 │ input_layer_1[0][0]        │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ input_layer (InputLayer)      │ (None, 2)                 │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ han22 (Dense)                 │ (None, 40)                │           2,040 │ han21[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ han1 (Dense)                  │ (None, 10)                │              30 │ input_layer[0][0]          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ han23 (Dense)                 │ (None, 30)                │           1,230 │ han22[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ han2 (Dense)                  │ (None, 20)                │             220 │ han1[0][0]                 │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ han24 (Dense)                 │ (None, 20)                │             620 │ han23[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ han3 (Dense)                  │ (None, 30)                │             630 │ han2[0][0]                 │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ han25 (Dense)                 │ (None, 10)                │             210 │ han24[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ han4 (Dense)                  │ (None, 5)                 │             155 │ han3[0][0]                 │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ han26 (Dense)                 │ (None, 3)                 │              33 │ han25[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ mg1 (Concatenate)             │ (None, 8)                 │               0 │ han4[0][0], han26[0][0]    │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ mg2 (Dense)                   │ (None, 10)                │              90 │ mg1[0][0]                  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ mg3 (Dense)                   │ (None, 5)                 │              55 │ mg2[0][0]                  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ last (Dense)                  │ (None, 1)                 │               6 │ mg3[0][0]                  │
└───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
 Total params: 5,519 (21.56 KB)
 Trainable params: 5,519 (21.56 KB)
 Non-trainable params: 0 (0.00 B)
"""

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit([x1_train,x2_train],y_train,epochs=100, batch_size=8)

#4. 평가, 예측
result = model.evaluate([x1_test,x2_test],y_test)
print("loss: ",result)

x1_pred = np.array([range(100,106),range(400,406)]).T
x2_pred = np.array([range(200,206),range(510,516),range(249,255)]).T
y_predict = model.predict([x1_pred,x2_pred])
print(y_predict)

# loss:  0.009885629639029503
# [[3099.1514]
#  [3100.65  ]
#  [3102.149 ]
#  [3103.6472]
#  [3105.179 ]
#  [3106.7634]]

# train_test_split 사용이후
# loss:  9.398062684340402e-05
# [[3099.996 ]
#  [3100.9993]
#  [3102.002 ]
#  [3103.0051]
#  [3104.0078]
#  [3105.0107]]