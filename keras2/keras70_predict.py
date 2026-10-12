import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input, LSTM, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os

mcp_path = './_save/keras70'
filename = os.listdir(mcp_path)[-1]
filepath = "".join([mcp_path,filename])

#1데이터
path = './_data/ensemble_data/'
amore_data = '아모레261012.csv'
hynix_data = '하이닉스261012.csv'
samsung_data = '삼성전자261012.csv'

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

#2.모델구성

#3.컴파일,훈련



y_predict= model.predict(x_test) 

y_predict = np.argmax(y_predict,axis=1) 
# print(y_predict)#[0 2 0 1 1 2 0 2 0 2 2 1 2 0 0 0 2 0 2 1 0 2 1 1 0 2 1 1 1 2]
y_test = np.argmax(y_test, axis=1)

accuracy_score =accuracy_score(y_test,y_predict)  
#지금까지는 y_predict 값은 [0.7,0.2,0.1]이런식으로 되어있어서 비교가 불가능함 >>가장큰 수를 1로 바꿔줘야함 그래서 결과를 [1,0,0]으로 변경후 비교 
print('acc_score :',accuracy_score)
