import pandas as pd
import numpy as np
import scipy
import tensorflow
import tqdm


df = pd.read_csv(
    "C:\\Users\\jamie\\Documents\\essential-coding-skills\\extra\\data.csv"
)
x = df.iloc[:, 1:-1]
lbls = df.iloc[:, -1]

print(lbls)

col1 = (df.iloc[:, 1] - df.iloc[:, 1].min()) / (
    df.iloc[:, 1].max() - df.iloc[:, 1].min()
)
col2 = (df.iloc[:, 2] - df.iloc[:, 2].min()) / (
    df.iloc[:, 2].max() - df.iloc[:, 2].min()
)
col3 = (df.iloc[:, 3] - df.iloc[:, 3].min()) / (
    df.iloc[:, 3].max() - df.iloc[:, 3].min()
)
col4 = (df.iloc[:, 4] - df.iloc[:, 4].min()) / (
    df.iloc[:, 4].max() - df.iloc[:, 4].min()
)

x = pd.concat([col1, col2, col3, col4, lbls], axis=1)
x = x.sample(frac=1)
lbls = x.iloc[:, -1]
x = x.iloc[:, 1:-1]

xtrain = x[:130]
ytrain = lbls[:130]

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
