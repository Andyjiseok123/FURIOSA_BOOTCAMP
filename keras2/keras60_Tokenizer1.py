from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np

text = '나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다.'

token = Tokenizer() # 객체(instance)

token.fit_on_texts([text])

print(token.word_index)     
#{'마구': 1, '진짜': 2, '매우': 3, '나는': 4, '지금': 5, '맛있는': 6, '김밥을': 7, '엄청': 8, '먹었다': 9}

print(token.word_counts)        
#OrderedDict([('나는', 1), ('지금', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('김밥을', 1), ('엄청', 1), ('마구', 4), ('먹었다', 1)])

x = token.texts_to_sequences([text])
print(x)
#[[4, 5, 2, 2, 3, 3, 6, 7, 8, 1, 1, 1, 1, 9]]
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