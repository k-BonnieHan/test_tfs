
import pandas as pd
import urllib.request
import matplotlib.pyplot as plt
import re
from konlpy.tag import Okt 
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_data = pd.read_table('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/ratings_train.txt')
test_data = pd.read_table('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/ratings_test.txt')

print('훈련용 리뷰 개수 :', len(train_data))  # 150000
print(train_data[:3])

print('테스트용 리뷰 개수 :', len(test_data))  # 50000
print(test_data[:3])

# 데이터 정제하기
print(train_data['document'].nunique(), train_data['label'].nunique()) # 중복 확인
# 총 150,000개의 샘플이 존재하는데 document열에서 중복을 제거한 샘플의 개수가 146,182개라는 것은 
# 약 4,000개의 중복 샘플이 존재한다는 의미입니다. label 열은 0 또는 1의 값만을 가지므로 2가 출력되었다. 

train_data.drop_duplicates(subset=['document'], inplace=True) # document 열에서 중복 제거
print('총 샘플의 수 :', len(train_data))  # 146183   

train_data['label'].value_counts().plot(kind = 'bar')
plt.show()   # 레이블의 분포가 균일한 것처럼 보인다.

print(train_data.groupby('label').size().reset_index(name = 'count')) 

# 리뷰 중에 Null 값을 가진 샘플이 있는지는 Pandas의 isnull().values.any()로 확인
print(train_data.isnull().values.any())  # True   데이터 중 Null 값을 가진 샘플이 있단 의미
print(train_data.isnull().sum())  # 어떤 열에 존재하는지 확인

# document 열에서 Null 값이 존재한다는 것을 조건으로 Null 값을 가진 샘플이 어느 인덱스의 위치에 존재하는지 한 번 출력
print(train_data.loc[train_data.document.isnull()])

# Null 값을 가진 샘플을 제거하겠다.
train_data = train_data.dropna(how = 'any') 
print(train_data.isnull().values.any()) 
print(len(train_data))  # 146182   

# 자음의 범위는 ㄱ ~ ㅎ, 모음의 범위는 ㅏ ~ ㅣ와 같이 지정할 수 있다.
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
print(train_data[:3])   # 한글과 공백을 제외하고 모두 제거 (온점과 같은 구두점 등은 제거) 
#          id                                           document  label
    # 0   9976970                                  아 더빙 진짜 짜증나네요 목소리      0
    # 1   3819312                         흠포스터보고 초딩영화줄오버연기조차 가볍지 않구나      1
    # 2  10265843                                  너무재밓었다그래서보는것을추천한다      0

# 네이버 영화 리뷰는 굳이 한글이 아니라 영어, 숫자, 특수문자로도 리뷰를 업로드할 수 있다. 
# train_data에 빈 값을 가진 행이 있다면 Null 값으로 변경하도록 하고, 다시 Null 값이 존재하는지 확인.
train_data['document'].replace('', np.nan, inplace=True)
print(train_data.isnull().sum())

print(train_data.loc[train_data.document.isnull()][:3]) # Null 값이 있는 행을 3개만 출력

# Null 샘플들은 아무런 의미도 없는 데이터므로 제거
train_data = train_data.dropna(how = 'any')
print(len(train_data))    # 145791

# 테스트 데이터에 지금까지 진행했던 전처리 과정들을 동일하게 진행한다.
test_data.drop_duplicates(subset = ['document'], inplace=True) 
test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") 
test_data['document'].replace('', np.nan, inplace=True)  
test_data = test_data.dropna(how='any')               
print('전처리 후 테스트용 샘플의 개수 :', len(test_data))        # 48995

# 토큰화 : 토큰화 과정에서 불용어를 제거하겠다. 불용어는 정의하기 나름.
stopwords = ['의','가','이','은','들','는','좀','잘','과','도','를','으로','자','에','와','한','하다']

# train_data에 형태소 분석기를 사용하여 토큰화를 하면서 불용어를 제거하여 X_train에 저장
okt = Okt()
X_train = []
for sentence in train_data['document']:
    temp_X = []
    temp_X = okt.morphs(sentence, stem=True) # 토큰화. 소요시간 좀 걸림
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
    X_train.append(temp_X)

print(X_train[:3])  # [['아', '더빙', '진짜', '짜증나다', '목소리'], ['흠', '포스터', ...

# 테스트 데이터에 대해서도 동일하게 토큰화.
X_test = []
for sentence in test_data['document']:
    temp_X = []
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
    X_test.append(temp_X)

print(X_test[:3])
# 지금까지 훈련 데이터와 테스트 데이터에 대해서 텍스트 전처리를 진행해보았다.

# 텍스트를 숫자로 처리할 수 있도록 훈련 데이터와 테스트 데이터에 정수 인코딩을 수행해야 한다. 
# 훈련 데이터에 대해서 단어 집합(vocaburary)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
# 이제 단어 집합이 생성되는 동시에 각 단어에 고유한 정수가 부여되었다.
print(tokenizer.word_index)    # {'영화': 1, '보다': 2, '을': 3, ... '수간': 43752}
# 단어가 43,000개가 넘게 존재한다. 각 정수는 전체 훈련 데이터에서 등장 빈도수가 높은 순서대로 부여되었기 때문에, 높은 정수가 부여된 단어들은 등장 빈도수가 매우 낮다는 것을 의미한다. 
# 여기서는 빈도수가 낮은 단어들은 자연어 처리에서 배제한다. 
# 등장 빈도수가 3회 미만인 단어들이 이 데이터에서 얼만큼의 비중을 차지하는지 확인해본다.
threshold = 3
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value
    if(value < threshold):# 단어의 등장 빈도수가 threshold보다 작으면
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt) * 100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq) * 100)

# 등장 빈도가 threshold 값인 3회 미만. 즉, 2회 이하인 단어들은 단어 집합에서 무려 절반 이상을 차지한다. 그래서 배제.
# 등장 빈도수가 2이하인 단어들의 수를 제외한 단어의 개수를 단어 집합의 최대 크기로 제한하겠다.
# 전체 단어 개수 중 빈도수 2이하인 단어 개수는 제거.  0번 패딩 토큰과 1번 OOV 토큰을 고려하여 +2
vocab_size = total_cnt - rare_cnt + 2
print('단어 집합의 크기 :', vocab_size)  # 19417

# 단어 집합의 크기는 19,417개다. 이를 케라스 토크나이저의 인자로 넘겨주면, 케라스 토크나이저는 텍스트 시퀀스를 숫자 시퀀스로 변환한다. 
# 이러한 정수 인코딩 과정에서 이보다 큰 숫자가 부여된 단어들은 OOV로 변환하겠다. 다시 말해 정수 1번으로 할당한다. 
tokenizer = Tokenizer(vocab_size, oov_token = 'OOV') 
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

print(X_train[:3]) # 정수 인코딩이 진행되었는지 확인
# [[51, 455, 17, 261, 660], [934, 458, 42, 603, 2, 215, 1450, 25, 962, 676, 20], [387, 2445, 2316, 5672, 3, 223, 10]]
# 각 샘플 내의 단어들은 각 단어에 대한 정수로 변환된 것을 확인할 수 있다. 
# 확인하지는 않겠지만, 이제 단어의 개수는 19,417개로 제한되었으므로 0번 단어 ~ 19,416번 단어까지만 사용한다. (0번 단어는 패딩을 위한 토큰, 1번 단어는 OOV를 위한 토큰이다.) 
# 다시 말해 19,417 이상의 정수는 더 이상 훈련 데이터에 존재하지 않는다.

# 이제 train_data에서 y_train과 y_test를 별도로 저장해준다.
y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])

# 빈 샘플들은 어떤 레이블이 붙어있던 의미가 없으므로 빈 샘플들을 제거해주는 작업을 하겠다. 
# 각 샘플들의 길이를 확인해서 길이가 0인 샘플들의 인덱스를 받아오겠다.
drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]

# 이제 drop_train에는 X_train으로부터 얻은 빈 샘플들의 인덱스가 저장되어 있다. 
# 앞서 훈련 데이터(X_train, y_train)의 샘플 갯수는 145,791개임을 확인했었다. 
# 그렇다면 빈 샘플들을 제거한 후의 샘플 갯수는 몇 개일까?
# 빈 샘플들을 제거
X_train = np.delete(X_train, drop_train, axis=0)
y_train = np.delete(y_train, drop_train, axis=0)
print(len(X_train))   # 145,380개로 샘플의 수가 줄어든 것을 확인
print(len(y_train))

print('리뷰의 최대 길이 :', max(len(l) for l in X_train))   # 72
print('리뷰의 평균 길이 :', sum(map(len, X_train))/len(X_train))
# 11.0021
plt.hist([len(s) for s in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

# 가장 긴 리뷰의 길이는 72이며, 그래프를 봤을 때 전체 데이터의 길이 분포는 대체적으로 약 11내외의 길이를 가지는 것을 볼 수 있다.
# 전체 샘플 중 길이가 max_len 이하인 샘플의 비율이 몇 %인지 확인하는 함수를 만든다.
def below_threshold_len(max_len, nested_list):
    cnt = 0
    for s in nested_list:
        if(len(s) <= max_len):
            cnt = cnt + 1
    print('전체샘플중 길이 %s 이하 샘플비율: %s'%(max_len, (cnt / len(nested_list))*100)) # 94.0830
# 위의 분포 그래프를 봤을 때, max_len = 30이 적당할 것 같다. 이 값이 얼마나 많은 리뷰 길이를 커버하는지 확인해본다.
max_len = 30
below_threshold_len(max_len, X_train)

# 전체 훈련 데이터 중 약 94%의 리뷰가 30이하의 길이를 가지는 것을 확인했다. 샘플의 길이 30으로 한다.
X_train = pad_sequences(X_train, maxlen = max_len)
X_test = pad_sequences(X_test, maxlen = max_len)

# LSTM으로 네이버 영화 리뷰 감성 분류하기 -----------------------
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# 임베딩 벡터의 차원은 100으로 정했고, 리뷰 분류를 위해서 LSTM을 사용

model = Sequential()
model.add(Embedding(vocab_size, 100))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

# 조기 종료(Early Stopping). 또한, ModelCheckpoint를 사용
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=2, save_best_only=True)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())

history = model.fit(X_train, y_train, epochs=10, callbacks=[es, mc], batch_size=64, validation_split=0.2)

loaded_model = load_model('best_model.h5') 
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1])) # 0.8544

# 리뷰 예측해보기
def sentiment_predict(new_sentence):
    new_sentence = okt.morphs(new_sentence, stem=True)    
    new_sentence = [word for word in new_sentence if not word in stopwords] 
    encoded = tokenizer.texts_to_sequences([new_sentence]) 
    pad_new = pad_sequences(encoded, maxlen = max_len)     
    score = float(loaded_model.predict(pad_new))         
    if(score > 0.5):
        print("{:.2f}% 확률로 긍정 리뷰\n".format(score * 100))
    else:
        print("{:.2f}% 확률로 부정 리뷰\n".format((1 - score) * 100))

sentiment_predict('이 영화 개꿀잼 ㅋㅋㅋ')
sentiment_predict('이 영화 핵노잼 ㅠㅠ')   
sentiment_predict('이딴게 영화냐 ㅉㅉ') 
sentiment_predict('감독 뭐하는 놈이냐?')


