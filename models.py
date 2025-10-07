from typing import List, Union
from encodings import undefined
from matplotlib.pylab import f
import numpy as np
from traitlets import Instance
import utils as ut

class DenseLayer:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2/input_dim)
        self.b = np.random.randn(1, output_dim) * 0.01 
        self.input_dim = input_dim
        self.output_dim = output_dim

    def params_count(self):
        return self.input_dim * self.output_dim + 1

class ActivationLayer:
    def __init__(self, method):
        if(method=="relu"):
            self.activate_func = self.relu
            self.activate_derivative = self.relu_derivative
        else:
            self.activate_func = None
            self.activate_derivative = None

    def forward(self, z):
        if(self.activate_func is not None):
            return self.activate_func(z)
        return z
    
    def backward(self, z):
        if(self.activate_func is not None):
            return self.activate_derivative(z)
        return z
    
    # ----------------------------- Method Part ----------------------------

    def relu(self, x):
        f = np.copy(x)
        f[x<0] = 0
        return f
    
    def relu_derivative(self, x):
        g = np.zeros_like(x)
        g[x > 0] = 1.0      # subgradient 0 at x==0
        return g

class Lr_Models:
    def __init__(self):
        # Using type hints.
        self.layers: List[Union[DenseLayer, ActivationLayer]] = []

    def add(self, layer):
        self.layers.append(layer)
    
    def add_layer(self, input_dim, output_dim):
        layer = DenseLayer(input_dim, output_dim)
        self.layers.append(layer)

    def add_ativation(self, method):
        layer = ActivationLayer(method)
        self.layers.append(layer)

    def total_params(self):
        return sum(layer.params_count() for layer in self.layer 
                      if isinstance(layer, DenseLayer))
    
    def predict(self, X):
        # Intial first a to forward
        a = X

        for layer in self.layers:
            # If now we're on dense layer
            if(isinstance(layer, DenseLayer)):
                if(a.shape[1] != layer.input_dim):
                    raise ValueError("Input dimension mismatch: expected " + 
                                     f"{layer.input_dim}, got {a.shape[1]}")
            
                # z = X * W + b
                # z = a ; no activate funciton
                a = np.dot(a, layer.W) + layer.b
            
            # If now we're on activation layer
            if(isinstance(layer, ActivationLayer)):
                # a = f(z) which now z = a
                a = layer.forward(a)

        # Return last layer output
        return a   

    def fit(self, X, y_true, epochs=1000, batch_size=32, learning_rate=0.01):   
        n_samples = X.shape[0]

        for epoch in range(epochs):
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)

                 # Fetch the batch
                y_true_batch = y_true[batch_start:batch_end]
                X_batch = X[batch_start:batch_end]

                # Forward pass
                a = X_batch
                al = [X_batch]  # to store a values for backpropagation
                zl = [X_batch]  # to store z values for backpropagation

                for layer in self.layers:
                    # If now we're on dense layer
                    if(isinstance(layer, DenseLayer)):
                        if(a.shape[1] != layer.input_dim):
                            raise ValueError("Input dimension mismatch: expected " + 
                                            f"{layer.input_dim}, got {a.shape[1]}")
            
                        # z = X * W + b
                        # z = a ; no activate funciton
                        a = np.dot(a, layer.W) + layer.b
                        zl.append(a)

                    # If now we're on activation layer
                    if(isinstance(layer, ActivationLayer)):
                        # a = f(z) which now z = a
                        a = layer.forward(a)
                        al.append(a)

                # Get zi
                y_pred_batch = a

                # dL/dz (MSE) = (2/N)*(zi-yi)
                dL_dz = (2/batch_size) * (y_pred_batch - y_true_batch)

                # Compute gradients and Update weights
                for i, layer in reversed(list(enumerate(self.layers))):  
                    if(isinstance(layer, DenseLayer)):                
                        # compute gradients w.r.t weights and biases
                        # dL/dW = dz/dw * dL/dz
                        # dL/dW = a^T * dL_dz
                        #
                        # dL/db = dz/dw * dL/dz
                        # dL/db = [1]^T * dL/dz
                        # dL/db = sum(dL/dz)

                        dL_dW = np.dot(al[i-1].T, dL_dz)
                        dL_db = np.sum(dL_dz, axis=0, keepdims=True)

                        # Update weights and biases
                        layer.W -= learning_rate * dL_dW
                        layer.b -= learning_rate * dL_db

                        # Not has activate function yet
                        dL_dz = np.dot(dL_dz, layer.W.T)

                    if(isinstance(layer, ActivationLayer)):
                        # compute dL_dz for the next layer (if any)
                        # z(l-1) -> a(l-1) -> z(l) -> L
                        # dL/da(l-1) = dz(l)/da(l-1) * dL/dz(l) 
                        # dL/da(l-1) =  dL/dz(l) * W^T
                        #
                        # dL/dz(l-1) = da(l-1)/dz(l-1) * dL/da(l-1)
                        # dL/da(l-q) = f'(z(l-1)) dot (dL/dz(l) * W^T)
                        dL_dz = self.relu_derivative(zl[i-1]) * dL_dz

        print(f"Epoch {epoch+1}/{epochs} mae: {ut.mae(y_true, self.predict(X))} completed.")