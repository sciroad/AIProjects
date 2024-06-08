import numpy as np  
import matplotlib.pyplot as plt

def y_function(x):
    return x**2

def y_derivative(x):
    return 2*x


x = np.arrange(-100, 100, 0.1)
y= y_function(x)

current_x = 100
learning_rate = 0.1

for i in range(100):
    plt.plot(x, y)
    plt.scatter(current_x, y_function(current_x))
    plt.show()
    current_x -= learning_rate * y_derivative(current_x)