from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np

text1 = '나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다.'
text2 = '개똥이는 기관사를 좋아한다. 말똥이는 잘생겼다. 길동이는 마구 마구 더 잘생겼다.'

token = Tokenizer() # 객체(instance)

token.fit_on_texts([text1,text2])

print(token.word_index)

print(token.word_counts)        

x = token.texts_to_sequences([text1,text2])
print(x)
x=np.concatenate((x[0],x[1]),axis=0)
# One hot encoding을 해야함

##################### 원핫 1. to_categorical ######################
#####################    from tensorflow   #######################
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)

##################### 원핫 2. pd.get_dummies ######################
#######################    from pandas   #########################
# y = pd.get_dummies(y,dtype=int)
# print(y)


####################### 원핫 3. sklearn.processing ######################
#######################    from sci-kit learn   #########################
from sklearn.preprocessing import OneHotEncoder
x = np.array(x).reshape(-1,1)
x = OneHotEncoder(sparse_output=False).fit_transform(x)
print(x)