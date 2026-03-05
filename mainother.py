import pandas as pd
import numpy as np
import scipy
import tensorflow
import tqdm


df = pd.read_csv("data.csv")
x = df.iloc[:, 1:-1]
lbls = df.iloc[:, -1]

print(lbls)


x = df.sample(frac=1)
lbls = x.iloc[:, -1]
x = x.iloc[:, 1:-1]

xtrain = x[:80]
ytrain = lbls[:80]

ytrain = pd.get_dummies(ytrain).values


import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *

m = Sequential()
m.add(Dense(10, input_dim=3, activation="tanh"))
m.add(Dense(40, activation="tanh"))
m.add(Dense(40, activation="tanh"))
m.add(Dense(3, activation="softmax"))


# m.add(Dense(10, input_dim=4, activation='relu'))
# m.add(Dense(20, activation='relu'))
# m.add(Dense(80, activation='relu'))
# m.add(Dense(3, activation='softmax'))

m.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

h = m.fit(xtrain, ytrain, epochs=100, batch_size=8, verbose=0)

print("Finished training!")

import matplotlib.pyplot as plt

plt.plot(h.history["loss"])
plt.title("Loss Curve")
plt.savefig("loss.png")
plt.show()


xtest = x[80:]
ytest = lbls[80:]
ytest = pd.get_dummies(ytest).values

loss, acc = m.evaluate(xtest, ytest)
print("Accuracy is", acc)
