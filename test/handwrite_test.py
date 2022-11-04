from keras.datasets import mnist
from neural_network import *

# Load training data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, -1)
x_test = x_test.reshape(10000, -1)

y_train = y_train.reshape(60000, -1)
y_test = y_test.reshape(10000, -1)

scalar = Scaler('min_max', 'one_hot')
scalar.x_fit(x_train)
scalar.y_fit(y_train)

x_train_nor = scalar.x_transform(x_train)
y_train_nor = scalar.y_transform(y_train)

model = Model()
model.add(Layer(200, inputs=784, activation='sigmoid'))
model.add(Layer(10, activation='linear'))
model.compile(optimizer='adam', metrics=['acc'])

tran_history = model.fit(x_train_nor[0:1000], y_train_nor[0:1000], 10)
model.plot_history()
