import numpy as np


f = lambda x: x**2 - 4*x + 6

def f(x):
    temp = x**2 - 4*x + 6
    return temp

gradient = lambda x: 2*x - 4

x = 500
epochs = 200
learning_rate = 0.25

print('step\t x\tf(x)')
print('{:02d} \t{:6.5f} \t{:6.5f}\t'.format(0,x,f(x)))

for i in range(epochs):
    x = x - learning_rate * gradient(x)
    # if abs(f(x)) > abs(f(x)):
    #     learning_rate *= 0.5
    # x = x_new
    print('{:02d} \t{:6.5f} \t{:6.5f}'.format(i+1,x,f(x)))
    

#     loss:  [2.959928035736084, 0.5519999861717224]
# acc:  0.552
