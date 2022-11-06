import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()

#Display training data
X, y = spiral_data( samples = 100 , classes = 3 )
plt.scatter(X[:, 0 ], X[:, 1 ], c = y, cmap = 'brg' )
plt.show()

#Define layer class
class Layer_Dense:
    #Class constructor init weights and biases
    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros(( 1 , n_neurons))
    
    # Calculate output values from inputs, weights and biases
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        
# ReLU activation
class Activation_ReLU :
    def forward ( self , inputs ):
        self.output = np.maximum( 0 , inputs)

# Softmax activation
class Activation_Softmax :
    # Forward pass
    def forward ( self , inputs ):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

# Common loss class
class Loss :
    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate ( self , output , y ):
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        # Return loss
        return data_loss

# Cross-entropy loss
class Loss_CategoricalCrossentropy ( Loss ):
    # Forward pass
    def forward ( self , y_pred , y_true ):
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7 , 1 - 1e-7 )
            
        # Probabilities for target values -
        # only if categorical labels
        if len (y_true.shape) == 1 :
            correct_confidences = y_pred_clipped[range ( len (y_pred_clipped)), y_true]
        # Mask values - only for one-hot encoded labels
        elif len (y_true.shape) == 2 :
            correct_confidences = np.sum(y_pred_clipped * y_true, axis = 1)
            
        neg_log = - np.log(correct_confidences)
        return neg_log
        
dense1 = Layer_Dense(2 , 3)
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values
dense2 = Layer_Dense( 3 , 3 )
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print (activation2.output[:5])

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)
print ('loss:', loss)

# Calculate accuracy from output of activation2 and targets
predictions = np.argmax(activation2.output, axis = 1)
if len (y.shape) == 2 :
    y = np.argmax(y, axis = 1 )
    
accuracy = np.mean(predictions == y)
print ( 'acc:' , accuracy)