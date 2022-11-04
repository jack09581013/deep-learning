import numpy as np

A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
C = np.random.randn(5, 6)

Y = np.zeros((3, 6), dtype='float')

for i in range(1000):
    n = Y.shape[0] * Y.shape[1]
    D = A.dot(B).dot(C)
    MSE_loss = 1 / n * (D - Y) ** 2
    delta_f_D = 2 / n * (D - Y)
    delta_B = A.T.dot(delta_f_D).dot(C.T)
    B -= delta_B * 0.01
    print(f'{MSE_loss.reshape(-1).sum():.2e}')

