import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#Load Dataset
mnist=tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test)=mnist.load_data()

#Normalization
x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)

plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()

#Build a Model
model=tf.keras.models.Sequential()

#Our images are 28x28 so to make the input layer flattend to 1*784
model.add(tf.keras.layers.Flatten())

#Adding layers
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

#Adding layer 2
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

#Output Layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

#Train the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

#To save the model
model.save('digit_detection.model')

#Load it back
new_model = tf.keras.models.load_model('digit_detection.model')

#To check if the model works
predictions = new_model.predict(x_test)
print(predictions)

print(np.argmax(predictions[15]))
plt.imshow(x_test[15],cmap=plt.cm.binary)
plt.show()

