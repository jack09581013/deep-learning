from optimizer import *

a = Adadelta()
x = 10
for i in range(300):
    x = a.get(x, 2*x)
    print(x)
