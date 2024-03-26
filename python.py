# creating the base python file 
import numpy as np
import matplotlib.pyplot as plt

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def relu_function(x):
    return np.maximum(0, x)

def leaky_relu_function(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def tanh_function(x):
    return np.tanh(x)

# Generating the data
x = np.linspace(-5, 5, 100)
y_sigmoid = sigmoid_function(x)
y_relu = relu_function(x)
y_leaky_relu = leaky_relu_function(x)
y_tanh = tanh_function(x)

# Plotting the graphs
plt.plot(x, y_sigmoid, label='Sigmoid')
plt.plot(x, y_relu, label='ReLU')
plt.plot(x, y_leaky_relu, label='Leaky ReLU')
plt.plot(x, y_tanh, label='Tanh')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Activation Functions')
plt.legend()
plt.grid(True)
plt.show()
