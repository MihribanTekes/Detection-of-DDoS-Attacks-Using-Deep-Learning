import keras
import keras.optimizers
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

model = Sequential():
model.add(Dense(128,input_dim=20, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='softmax'))
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(scaled_train_samples, train_labels, batch_size=10, epochs=20, shuffle=True, verbose=2)

data = pd.read_csv("47.txt")
data.drop(data.columns[[0,1,8,9,10,12,13,14,15,16,17,18,19,20,21,23,26,27,32,34,39,40,42,43,44,45
,46]], axis=1, inplace=True)

le = LabelEncoder()

data['protocol_type'] = le.fit_transform(data['protocol_type'])
data['service'] = le.fit_transform(data['service'])
data['flag'] = le.fit_transform(data['flag'])

values = data.iloc[:,0:20].values
labels = data.iloc[:,20].values

scaler = MinMaxScaler(feature_range=(0, 1))
rescaledValues = scaler.fit_transform(values)

predictions = model.predict(rescaledValues, batch_size=100, verbose=1)

for i in predictions:
print(i)
