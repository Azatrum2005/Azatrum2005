import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

fashion_mnist=keras.datasets.fashion_mnist
(trainimg,trainl),(testimg,testl)=fashion_mnist.load_data()

#plt.imshow(trainimg[5],cmap="gray",vmin=0,vmax=255)
#plt.show()
model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    #keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(units=10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.optimizers.Adam(),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(trainimg,trainl,epochs=1)
testloss=model.evaluate(testimg,testl)
pred=model.predict(testimg)
n=10000
for i in range(n):
    print(pred[i])
    print(testl[i])
    print(np.where(pred[i]==max(pred[i]))[0])
    a=input("enter")
    if True:
        continue