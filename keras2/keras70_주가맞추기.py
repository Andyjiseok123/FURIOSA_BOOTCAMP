# 삼성전자 주가 (10/13 시가) 맞추기
# ensemble 모델 사용할것
# LSTM 및 Conv1D 사용할것

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM, Conv1D, Flatten
import time
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import datetime

mcp_path = './_save/keras70/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:4f}.keras'
filepath = "".join([mcp_path,"k31_",date,filename])

# split 함수
def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset)-size+1):
        subset = dataset[i:(i+size)]
        aaa.append(subset)
    return np.array(aaa)


path = './_data/ensemble_data/'
amore_data = '아모레261012.csv'
hynix_data = '하이닉스261012.csv'
samsung_data = '삼성전자261012.csv'

#1. 데이터
# pandas로 읽어오기
df_amore = pd.read_csv(path+amore_data,encoding='cp949',index_col=0).drop('대비',axis=1).rename(columns={'Unnamed: 6':'대비'})
df_hynix = pd.read_csv(path+hynix_data,encoding='cp949',index_col=0).drop('대비',axis=1).rename(columns={'Unnamed: 6':'대비'})
df_samsung = pd.read_csv(path+samsung_data,encoding='cp949',index_col=0).drop('대비',axis=1).rename(columns={'Unnamed: 6':'대비'})
# print(type(df_amore))       #pandas.DaraFrame
# print(df_amore.shape,df_hynix.shape,df_samsung.shape)       # (2060, 15) (1300, 15) (1120, 15)

# 1열씩 읽으면서 type이 str일때 콤마를 제거하고 int타입으로 변경
# print(len(df_amore.iloc[0]))            #15
for i in range(len(df_amore.iloc[0])):
    if df_amore[df_amore.columns[i]].dtype == 'str':
        df_amore_Temp = pd.to_numeric(df_amore[df_amore.columns[i]].str.replace(',',''))
        df_amore[df_amore.columns[i]] = df_amore_Temp
    else:
        pass

for i in range(len(df_hynix.iloc[0])):
    if df_hynix[df_hynix.columns[i]].dtype == 'str':
        df_amore_Temp = pd.to_numeric(df_hynix[df_hynix.columns[i]].str.replace(',',''))
        df_hynix[df_hynix.columns[i]] = df_amore_Temp
    else:
        pass

for i in range(len(df_samsung.iloc[0])):
    if df_samsung[df_samsung.columns[i]].dtype == 'str':
        df_amore_Temp = pd.to_numeric(df_samsung[df_samsung.columns[i]].str.replace(',',''))
        df_samsung[df_samsung.columns[i]] = df_amore_Temp
    else:
        pass

# print(df_amore.shape,df_hynix.shape,df_samsung.shape)       # (2060, 15) (1300, 15) (1120, 15)

# 제일 짧은 samsung의 데이터 기준으로 slicing 및 오늘 데이터 삭제
slice_length = min(df_amore.shape[0],df_hynix.shape[0],df_samsung.shape[0]) #1120
df_amore = df_amore[1:slice_length]
df_hynix = df_hynix[1:slice_length]
df_samsung = df_samsung[1:slice_length]
# print(type(df_samsung))

# target(삼성전자 시가) 추출
y_target = df_samsung['시가']
df_samsung = df_samsung.drop(['시가'],axis=1)

# 모든 데이터를 numpy array행렬 변환
df_amore = df_amore.to_numpy()
df_hynix = df_hynix.to_numpy()
df_samsung = df_samsung.to_numpy()
y_target = y_target.to_numpy()



# train_test_split
lstm_train_amore, lstm_test_amore, lstm_train_hynix, lstm_test_hynix, lstm_train_samsung, lstm_test_samsung, conv_train_amore, conv_test_amore, conv_train_hynix, conv_test_hynix, conv_train_samsung, conv_test_samsung, y_train, y_test = train_test_split(df_amore,df_hynix,df_samsung,df_amore,df_hynix,df_samsung,y_target,train_size=0.85,random_state=1214423)

# LSTM용 데이터 split
time_step = 15

# 모든 데이터에서 15일 시퀀스 생성
lstm_data_amore = split_x(df_amore, time_step)
lstm_data_hynix = split_x(df_hynix, time_step)
lstm_data_samsung = split_x(df_samsung, time_step)

# 시퀀스의 마지막 날짜에 해당하는 시가를 타깃으로 설정
y_data = y_target[time_step - 1:]

# 입력 데이터의 샘플 수를 타깃과 일치시킴
assert len(lstm_data_amore) == len(y_data)
assert len(lstm_data_hynix) == len(y_data)
assert len(lstm_data_samsung) == len(y_data)

# Conv1D도 동일한 3차원 시퀀스 사용
conv_data_amore = lstm_data_amore.copy()
conv_data_hynix = lstm_data_hynix.copy()
conv_data_samsung = lstm_data_samsung.copy()

# 모든 입력과 타깃을 동일한 인덱스로 무작위 분할
(
    lstm_train_amore,
    lstm_test_amore,
    lstm_train_hynix,
    lstm_test_hynix,
    lstm_train_samsung,
    lstm_test_samsung,
    conv_train_amore,
    conv_test_amore,
    conv_train_hynix,
    conv_test_hynix,
    conv_train_samsung,
    conv_test_samsung,
    y_train,
    y_test
) = train_test_split(
    lstm_data_amore,
    lstm_data_hynix,
    lstm_data_samsung,
    conv_data_amore,
    conv_data_hynix,
    conv_data_samsung,
    y_data,
    train_size=0.85,
    random_state=1214423
)
#2. 모델 구성

# amore LSTM
input1 = Input(shape=(15,15))
lstm1_1 = LSTM(20,name='lstm_amore1')(input1)
dense1_1 = Dense(20,activation='relu',name='lstm_amore2')(lstm1_1)
output1_1 = Dense(20,activation='relu',name='lstm_amore3')(dense1_1)
# model1 = Model(inputs=input1, outputs=output1)            #merge할때 필요가 없음

# amore Conv1D
input2 = Input(shape=(15,15))
conv2_1 = Conv1D(filters=10, kernel_size=5)(input2)
conv2_2 = Conv1D(10, 2)(conv2_1)
flatten1_1 = Flatten()(conv2_2)
dense2_1 = Dense(10,activation='relu')(flatten1_1)
output2_1 = Dense(10,activation='relu')(dense2_1)

# hynix LSTM
input3 = Input(shape=(15,15))
lstm3_1 = LSTM(10)(input3)
dense3_1 = Dense(20,activation='relu')(lstm3_1)
output3_1 = Dense(20,activation='relu')(dense3_1)
# model1 = Model(inputs=input1, outputs=output1)            #merge할때 필요가 없음

# hynix Conv1D
input4 = Input(shape=(15,15))
conv4_1 = Conv1D(filters=10, kernel_size=5)(input4)
conv4_2 = Conv1D(10, 2)(conv4_1)
flatten4_1 = Flatten()(conv4_2)
dense4_1 = Dense(10,activation='relu')(flatten4_1)
output4_1 = Dense(10,activation='relu')(dense4_1)

# samsung LSTM
input5 = Input(shape=(15,14))
lstm5_1 = LSTM(10)(input5)
dense5_1 = Dense(20,activation='relu')(lstm5_1)
output5_1 = Dense(10,activation='relu')(dense5_1)
# model1 = Model(inputs=input1, outputs=output1)            #merge할때 필요가 없음

# samsung Conv1D
input6 = Input(shape=(15,14))
conv6_1 = Conv1D(filters=10, kernel_size=5)(input6)
conv6_2 = Conv1D(10, 2)(conv6_1)
flatten6_1 = Flatten()(conv6_2)
dense6_1 = Dense(10,activation='relu')(flatten6_1)
output6_1 = Dense(10,activation='relu')(dense6_1)

# amore 모델 합치기
from keras.layers import concatenate, Concatenate
merge1_1 = concatenate([output1_1, output2_1])
merge1_2 = Dense(10,activation='relu')(merge1_1)

# hynix 모델 합치기
merge2_1 = concatenate([output3_1, output4_1])
merge2_2 = Dense(10,activation='relu')(merge2_1)

# samsung 모델 합치기
merge3_1 = concatenate([output5_1, output6_1])
merge3_2 = Dense(10,activation='relu')(merge3_1)

# 두번째 레이어 모델
merge_last = concatenate([merge1_2,merge2_2,merge3_2])
input41 = Dense(10,activation='relu')(merge_last)
dense41 = Dense(20,activation='relu')(input41)
last_output1 = Dense(1)(dense41)

model = Model(inputs=[input1,input2,input3,input4,input5,input6],outputs=[last_output1])

# model.summary()

#3. 컴파일, 훈련


model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss',mode='min',patience=100, restore_best_weights=True, )
mcp = ModelCheckpoint(monitor='val_loss',mode='min',save_best_only=True,filepath=filepath)
rlr = ReduceLROnPlateau(monitor='val_loss',mode='auto',patience=50,factor=0.5,)


model.fit([lstm_train_amore,conv_train_amore,lstm_train_hynix,conv_train_hynix,lstm_train_samsung,conv_train_samsung],y_train,
          epochs=10000, batch_size=8, callbacks=[es,mcp,rlr], validation_split=15/85)

#4. 평가, 예측
result = model.evaluate([lstm_test_amore,conv_test_amore,lstm_test_hynix,conv_test_hynix,lstm_test_samsung,conv_test_samsung],y_test)
print("loss: ",result)
