import pandas as pd
import numpy as np
import scipy
import tensorflow
import tqdm


df = pd.read_csv(
    "C:\\Users\\jamie\\Documents\\essential-coding-skills\\extra\\data.csv"
)

features = df.iloc[:, 1:-1]
labels = df.iloc[:, -1]

print(labels)

def create_cols(df, col_index):
    col = (df.iloc[:, col_index] - df.iloc[:, col_index].min()) / (
        df.iloc[:, col_index].max() - df.iloc[:, col_index].min()
    )
    return col

cols = []
for i in range(1, 5):
    cols.append(create_cols(df, i))

x = pd.concat(cols + [labels], axis=1)
x = x.sample(frac=1) 
labels = x.iloc[:, -1]
x = x.iloc[:, 1:-1]

xtrain = x[:130]
ytrain = labels[:130]

print("X train:")
print(xtrain)

ytrain = pd.get_dummies(ytrain).values
print("Y train:")
print(ytrain)


import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *

m = Sequential()
m.add(Dense(10, input_dim=3, activation="tanh"))
# add a hidden layer using the relu activation
m.add(Dense(8, activation="tanh"))
m.add(Dense(3, activation="softmax"))


# m.add(Dense(10, input_dim=4, activation='relu'))
# m.add(Dense(20, activation='relu'))
# m.add(Dense(80, activation='relu'))
# m.add(Dense(3, activation='softmax'))

m.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

h = m.fit(xtrain, ytrain, epochs=30, batch_size=15, verbose=0)

print("Finished training!")

import matplotlib.pyplot as plt

plt.plot(h.history["loss"])
plt.title("Loss Curve")
plt.savefig("loss.png")
plt.show()


xtest = x[130:]
ytest = lbls[130:]
ytest = pd.get_dummies(ytest).values

loss, acc = m.evaluate(xtest, ytest)
print("Accuracy is", acc)
