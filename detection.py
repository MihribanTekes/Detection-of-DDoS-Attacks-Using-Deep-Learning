# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 00:53:38 2017

@author: mihriban
"""

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



def create_model():
    optimizer = keras.optimizers.Adam()
    model = Sequential()
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
    return model

seed = 7
np.random.seed(seed)



data = pd.read_csv("KDDTrain+.txt")
data.drop(data.columns[[0,8,9,10,12,13,14,15,16,17,18,19,20,21,23,26,27,32,34,39,40,42]], axis=1, inplace=True)


le = LabelEncoder()

data['protocol_type'] = le.fit_transform(data['protocol_type'])
data['service'] = le.fit_transform(data['service'])
data['flag'] = le.fit_transform(data['flag'])
data['a_class'] = le.fit_transform(data['a_class'])


values = data.iloc[:,0:20].values
labels = data.iloc[:,20].values

scaler = MinMaxScaler(feature_range=(0, 1))
rescaledValues = scaler.fit_transform(values)

model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=1)
kfold = StratifiedKFold(n_splits=10,random_state=seed)
results = cross_val_score(model, rescaledValues, labels, cv=kfold)
model.save('my_model.h5')
print(results.mean())
