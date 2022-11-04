from sklearn.datasets import load_iris
from neural_network import *

data = load_iris()
X = data['data']
Y = data['target'].reshape(-1, 1)

scalar = Scaler('min_max', 'one_hot')
scalar.x_fit(X)
scalar.y_fit(Y)

X_nor = scalar.x_transform(X)
Y_nor = scalar.y_transform(Y)

model = Model()
model.add(Layer(4, inputs=4, activation='sigmoid'))
model.add(Layer(4, activation='sigmoid'))
model.add(Layer(3, activation='linear'))
model.compile(optimizer='adam', metrics=['acc'])

tran_history = model.fit(X_nor, Y_nor, 40)
model.plot_history()


