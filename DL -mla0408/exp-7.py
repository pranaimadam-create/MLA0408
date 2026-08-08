import numpy as np
import matplotlib.pyplot as plt

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Generate values
x = np.arange(-5, 5, 0.1)

# Plot sigmoid function
plt.plot(x, sigmoid(x), color='pink')

plt.title('Visualization of the Sigmoid Function')
plt.xlabel('Input (z)')
plt.ylabel('Sigmoid Output')
plt.grid()
plt.show()
