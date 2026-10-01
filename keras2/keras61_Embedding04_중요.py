import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

#1. 데이터
docs = [
    '너무 재미있다.', '참 최고에요', '참 잘만든 영화에요',
    '추천하고 싶은 영화입니다.', '한 번 더 보고 싶어요', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다', '참 재밌네요',
    '개똥이 바보', '말똥이 잘생겼다', '길동이 또 구라친다'
]

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])

token = Tokenizer()
token.fit_on_texts(docs)
# print(token.word_index)
#{'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5, 
# '영화에요': 6, '추천하고': 7, '싶은': 8, '영화입니다': 9, '한': 10, '번': 11, 
# '더': 12, '보고': 13, '싶어요': 14, '글쎄': 15, '별로에요': 16, '생각보다': 17, '지루해요': 18, 
# '연기가': 19, '어색해요': 20, '재미없어요': 21, '재미없다': 22, '재밌네요': 23, '개똥이': 24, 
# '바보': 25, '말똥이': 26, '잘생겼다': 27, '길동이': 28, '또': 29, '구라친다': 30}
x = token.texts_to_sequences(docs)
# print(x)
# [[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14], [15], [16], [17, 18], [19, 20], [21], [2, 22], [1, 23], [24, 25], [26, 27], [28, 29, 30]]

######################## 패딩 ##############################
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_x = pad_sequences(x,
                        padding='pre',          #뒤에 채우려면 post
                        maxlen = 5,     
                        truncating='pre'        #길면  default로 앞에서 자름
)
print(padded_x.shape)                           #(15,5)

#2. 모델
from keras.layers import Embedding, SimpleRNN
"""
###################### 임베딩 1 ######################
model = Sequential()
model.add(Embedding(input_dim=30, output_dim=10, input_length=5))
# input_dim = 단어사전의 갯수, output_dim = 차원
model.add(SimpleRNN(units=10))
model.add(Dense(1))
model.summary()
"""
###################### 임베딩 2 ######################
model = Sequential()
model.add(Embedding(input_dim=30, output_dim=10))               #input_length를 명시하지 않아도 됨, parameter의 이름도 필요가 없음 대신 순서 조심
# input_dim = 단어사전의 갯수, output_dim = 차원
model.add(SimpleRNN(units=10))
model.add(Dense(1))
model.summary()

#3.
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(padded_x,labels,epochs=100)

