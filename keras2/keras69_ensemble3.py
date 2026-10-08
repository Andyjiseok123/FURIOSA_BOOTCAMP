import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input
import time
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

#1. 데이터
x1_datasets = np.array([range(100),range(301,401)]).T
x2_datasets = np.array([range(101,201),range(411,511),range(150,250)]).transpose()
x3_datasets = np.array([range(100),range(301,401),range(77,177),range(33,133)]).T

y1 = np.array(range(3001,3101))
y2 = np.array(range(13001,13101))

x1_train,x1_test,x2_train,x2_test,x3_train,x3_test,y1_train,y1_test,y2_train,y2_test = train_test_split(
    x1_datasets,x2_datasets,x3_datasets,y1,y2,train_size=0.85,random_state=414)
# 순서확인할것 
# print(x1_train.shape)           # (85,2)
# print(x2_train.shape)           # (85,3)
# print(x1_test.shape)            # (15,2)
# print(x2_test.shape)            # (15,3)
# print(y_train.shape)            # (85,)
# print(y_test.shape)             # (15,)
# exit()
# 3-2 layer 모델 만들어보기
#2-1 모델
input1 = Input(shape=(2,))
dense1 = Dense(10,activation='relu',name='han1')(input1)
dense2 = Dense(20,activation='relu',name='han2')(dense1)
dense3 = Dense(30,activation='relu',name='han3')(dense2)
output1 = Dense(4,activation='relu',name='han4')(dense3)
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

#2-3 모델
input31 = Input(shape=(4,))
dense31 = Dense(50,name='han31')(input31)
dense32 = Dense(40,name='han32')(dense31)
dense33 = Dense(30,name='han33')(dense32)
output31 = Dense(3,name='han36')(dense33)

#2-4 모델 합치기
from keras.layers import concatenate, Concatenate
merge1 = concatenate([output1, output21],name='mg1')
merge2 = Dense(10, name='mg2')(merge1)
merge3 = Dense(10, name='mg3')(merge2)

#2-5 두번째 레이어 모델
input41 = Dense(10,activation='relu')(merge3)
dense41 = Dense(20,activation='relu')(input41)
dense42 = Dense(20,activation='relu')(dense41)
last_output1 = Dense(1)(dense41)

input51 = Dense(10,activation='relu')(merge3)
last_output2 = Dense(1)(input51)

model = Model(inputs=[input1,input21,input31],outputs=[last_output1,last_output2])

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
│ han4 (Dense)                  │ (None, 4)                 │             124 │ han3[0][0]                 │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ han26 (Dense)                 │ (None, 3)                 │              33 │ han25[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ mg1 (Concatenate)             │ (None, 7)                 │               0 │ han4[0][0], han26[0][0]    │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ mg2 (Dense)                   │ (None, 10)                │              80 │ mg1[0][0]                  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ mg3 (Dense)                   │ (None, 10)                │             110 │ mg2[0][0]                  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense (Dense)                 │ (None, 10)                │             110 │ mg3[0][0]                  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_1 (Dense)               │ (None, 20)                │             220 │ dense[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_4 (Dense)               │ (None, 10)                │             110 │ mg3[0][0]                  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ input_layer_2 (InputLayer)    │ (None, 4)                 │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_3 (Dense)               │ (None, 1)                 │              21 │ dense_1[0][0]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_5 (Dense)               │ (None, 1)                 │              11 │ dense_4[0][0]              │
└───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
 Total params: 5,999 (23.43 KB)
 Trainable params: 5,999 (23.43 KB)
 Non-trainable params: 0 (0.00 B)
"""
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit([x1_train,x2_train,x3_train],[y1_train,y2_train],epochs=500, batch_size=8)

#4. 평가, 예측
result = model.evaluate([x1_test,x2_test,x3_test],[y1_test,y2_test])
print("loss: ",result)
# x1_datasets = np.array([range(100),range(301,401)]).T
x1_pred = np.array([range(100,106),range(400,406)]).T
# x2_datasets = np.array([range(101,201),range(411,511),range(150,250)]).transpose()
x2_pred = np.array([range(200,206),range(510,516),range(249,255)]).T
# x3_datasets = np.array([range(100),range(301,401),range(77,177),range(33,133)]).T
x3_pred = np.array([range(100,106),range(400,406),range(176,182),range(132,138)]).T
y_predict = model.predict([x1_pred,x2_pred,x3_pred])
print(y_predict)

# loss:  [26.3951416015625, 5.96912956237793, 20.42601203918457]
# [array([[3095.2437],
#        [3096.0312],
#        [3096.8213],
#        [3097.6104],
#        [3098.4004],
#        [3099.1895]], dtype=float32), 
#  array([[13098.301],
#        [13099.84 ],
#        [13101.372],
#        [13102.9  ],
#        [13104.432],
#        [13105.959]], dtype=float32)]