# iris dataset으로 분류. ROC Curve 출력.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

iris = load_iris()
#print(iris.DESCR)
x = iris.data
print(x[:2])
y = iris.target
print(y)
names = iris.target_names
feature_names = names
print(feature_names)

# label 원핫인코딩. keras:to_categorical, sklearn:OneHotEncoder, numpy:np.eye(), pandas:get_dummies()...
onehot = OneHotEncoder(categories = 'auto')
print(onehot)
y = onehot.fit_transform(y[:, np.newaxis]).toarray()
print(y[:5], y.shape)

# feature 표준화
scaler = StandardScaler()
x_scale = scaler.fit_transform(x)
print(x_scale[:2])

# train / test
x_train,x_test,y_train,y_test = train_test_split(x_scale, y, test_size = 0.3, random_state = 1)

n_features = x_train.shape[1]
n_classes = y_train.shape[1]
print(n_features, ' ', n_classes)  # 4   3
#------------------------------------------

# n의 갯수 만큼 모델 생성을 위한 함수
def create_custom_model(input_dim, output_dim, out_nodes, n, model_name='model'):
    def create_model():
        model = Sequential(name = model_name)
        for _ in range(n):
            model.add(Dense(out_nodes, input_dim = input_dim, activation='relu'))  # 은닉층
        
        model.add(Dense(output_dim, activation='softmax'))   # 출력층
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        return model
    return create_model

models = [create_custom_model(n_features, n_classes, 10, n, 'model_{}'.format(n)) for n in range(1, 4)]

for create_model in models:
    print('@@@')
    create_model().summary()


