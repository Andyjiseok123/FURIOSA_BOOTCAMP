import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input
import time
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

#1. 데이터
x1_datasets = np.array([range(100),range(301,401)]).T
# x2_datasets = np.array([range(101,201),range(411,511),range(150,250)]).transpose()
# x3_datasets = np.array([range(100),range(301,401),range(77,177),range(33,133)]).T

y1 = np.array(range(3001,3101))
y2 = np.array(range(13001,13101))

x1_train,x1_test,y1_train,y1_test,y2_train,y2_test = train_test_split(
    x1_datasets,y1,y2,train_size=0.85,random_state=414)
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


#2-5 두번째 레이어 모델
input41 = Dense(10,activation='relu')(output1)
dense41 = Dense(20,activation='relu')(input41)
dense42 = Dense(20,activation='relu')(dense41)
last_output1 = Dense(1)(dense41)

input51 = Dense(10,activation='relu')(output1)
last_output2 = Dense(1)(input51)

model = Model(inputs=[input1],outputs=[last_output1,last_output2])

model.summary()
"""
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ input_layer (InputLayer)      │ (None, 2)                 │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ han1 (Dense)                  │ (None, 10)                │              30 │ input_layer[0][0]          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ han2 (Dense)                  │ (None, 20)                │             220 │ han1[0][0]                 │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ han3 (Dense)                  │ (None, 30)                │             630 │ han2[0][0]                 │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ han4 (Dense)                  │ (None, 4)                 │             124 │ han3[0][0]                 │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense (Dense)                 │ (None, 10)                │              50 │ han4[0][0]                 │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_1 (Dense)               │ (None, 20)                │             220 │ dense[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_4 (Dense)               │ (None, 10)                │              50 │ han4[0][0]                 │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_3 (Dense)               │ (None, 1)                 │              21 │ dense_1[0][0]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_5 (Dense)               │ (None, 1)                 │              11 │ dense_4[0][0]              │
└───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
 Total params: 1,356 (5.30 KB)
 Trainable params: 1,356 (5.30 KB)
 Non-trainable params: 0 (0.00 B)
"""
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit([x1_train],[y1_train,y2_train],epochs=500, batch_size=8)

#4. 평가, 예측
result = model.evaluate([x1_test],[y1_test,y2_test])
print("loss: ",result)
# x1_datasets = np.array([range(100),range(301,401)]).T
x1_pred = np.array([range(100,106),range(400,406)]).T
y_predict = model.predict([x1_pred])
print(y_predict)

# loss:  [355.4035949707031, 335.1668701171875, 20.236696243286133]
# [array([[3061.0522],
#        [3061.3262],
#        [3061.5996],
#        [3061.873 ],
#        [3062.1467],
#        [3062.421 ]], dtype=float32), array([[13063.811 ],
#        [13064.979 ],
#        [13066.146 ],
#        [13067.3125],
#        [13068.481 ],
#        [13069.651 ]], dtype=float32)]