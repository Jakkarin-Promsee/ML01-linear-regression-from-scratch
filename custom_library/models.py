from typing import List, Union
import matplotlib.pyplot as plt
import numpy as np
from zmq import NULL
from . import utils as ut

class DenseLayer:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2/input_dim)
        self.b = np.random.randn(1, output_dim) * 0.01 
        self.input_dim = input_dim
        self.output_dim = output_dim

    def params_count(self):
        return self.input_dim * self.output_dim + self.output_dim

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
        if(self.activate_derivative is not None):
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

class History:
    def __init__(self):
        # Each entry = [train_acc, val_acc]
        self.history = []

        # Each entry = [pred, y_true]
        self.last_predict = []
        self.last_X = []
       
        # Save best weight
        self.layers = []
        self.best_layers = []
        self.val_acc = None

    def save_model(self, layers, val_acc):
        """Save model weights"""
        self.layers = layers

        if(self.val_acc == None):
            self.best_layers = layers
            self.val_acc = val_acc
            return

        if(self.val_acc > val_acc):
            self.best_layers = layers
            self.val_acc = val_acc
    
    def get_layers(self):
        return self.layers
    
    def get_best_layers(self):
        return self.best_layers

    def get_best_loss(self):
        return self.val_acc

    def save(self, train_acc, val_acc):
        """Store training and validation accuracy per epoch"""
        self.history.append(np.array([train_acc, val_acc]))

    def save_predict(self, X, predict, y_true):
        """Store predictions and corresponding true values"""
        predict = np.array(predict).flatten()
        y_true = np.array(y_true).flatten()
        
        self.last_X = X
        self.last_predict = [predict, y_true]

    def display_trend(self, ref="x", axis=0, sort=True, show_X=False):
        """Visualize the trend of predictions vs true values. ref="x"||"y", x for input data set, y for expect data set"""
        if not self.last_predict:
            print("No predictions to display.")
            return

        X = np.asarray(self.last_X)
        y_pred, y_true = self.last_predict

        if sort:
            if(ref=="y"):
                sort_idx = np.argsort(y_true)
            else:
                sort_idx = np.argsort(X if X.ndim == 1 else X[:, axis])

            X = X[sort_idx]
            y_true = y_true[sort_idx]
            y_pred = y_pred[sort_idx]

        i = np.arange(len(y_true))
        if(show_X):
            plt.scatter(i, X if X.ndim == 1 else X[:, axis], alpha=0.5, color='yellow', label='T')

        plt.scatter(i, y_pred, alpha=0.4, color='red', label='Predictions')
        plt.scatter(i, y_true, alpha=0.4, color='blue', label='True Values')
        plt.title("Prediction Trend")
        plt.xlabel("Sample Index (sorted by true value)")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

    def evaluate(self, start=0, end=None):
        """Plot training and validation accuracy history"""
        if not self.history:
            print("No training history available.")
            return

        hist_arr = np.array(self.history)
        if end is None:
            end = len(hist_arr)

        train_acc = hist_arr[start:end, 0]
        val_acc = hist_arr[start:end, 1]

        plt.plot(train_acc, label='Train Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training History')
        plt.legend()
        plt.show()

        y_pred, y_true = self.last_predict
        plt.scatter(y_true, y_pred)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.plot([y_true.min(), y_true.max()], [y_pred.min(), y_pred.max()], 'r--')
        plt.title('True vs Predicted Values')
        plt.show()

class Lr_Models:
    def __init__(self):
        self.history = History()

        # Using type hints.
        self.layers: List[Union[DenseLayer, ActivationLayer]] = []
        self.last_output_dim = NULL
    
    def add(self, dense, activation="", input_shape=NULL):
        """Add layer to models. First adding should provide input_shape"""
        # Set input_shape
        if(self.last_output_dim == NULL and input_shape == NULL):
            raise ValueError("Input shape missing")
        elif(self.last_output_dim == NULL):
            self.last_output_dim = input_shape

        # Create Dense layer
        layer = DenseLayer(self.last_output_dim, dense)
        self.layers.append(layer)
        self.last_output_dim = dense

        # Create Activation layer
        if(activation!=""):
            layer = ActivationLayer(activation)
            self.layers.append(layer)

    def add_activation(self, method):
        """Add activation layer"""
        layer = ActivationLayer(method)
        self.layers.append(layer)

    def load(self, history=History()):
        """"""
        self.history = history
        self.layers = history.get_layers()

    def total_params(self):
        """Get totoal parameters (int)"""
        return sum(layer.params_count() for layer in self.layers if isinstance(layer, DenseLayer)) 
    
    
    def predict_best(self, X):
        self.layers = self.history.get_best_layers()
        pred = self.predict(X)
        self.layers = self.history.get_layers()

        return pred
    
    def predict(self, X):
        """Get prediction from single point / array point"""
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

    def fit(self, X_train, y_train, X_eval, y_eval, epochs=1000, batch_size=32, learning_rate=0.01):
        """train models"""
        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)

                 # Fetch the batch
                y_true_batch = y_train[batch_start:batch_end]
                X_batch = X_train[batch_start:batch_end]

                # Forward pass
                a = np.array(X_batch)
                zal = [a] # to keep both z or a


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

                    # keep forward data
                    zal.append(a)

                # Get zi
                y_pred_batch = a

                # dL/dz (MSE) = (2/N)*(zi-yi)
                dL_dz = (2/batch_size) * (y_pred_batch - y_true_batch)

                # Set iterator
                i = len(zal)-1 

                # Compute gradients and Update weights
                for layer in reversed(list(self.layers)): 
                    # cause i=max() have done calculate at first dL/dz
                    i-=1

                    if(isinstance(layer, DenseLayer)):                
                        # z(l) = a(l-1) * W(l) + b(l)
                        # a(l-1) -> z(l) -> a(l) -> L
                        # compute gradients w.r.t weights and biases
                        # dL/dW = dz/dw * dL/dz
                        # dL/dW = a^T * dL_dz
                        #
                        # dL/db = dz/db * dL/dz
                        # dL/db = [1]^T * dL/dz
                        # dL/db = sum(dL/dz)
                        #
                        # dL/dw(l-1) = dz/dw(l-1) * dL/dz(l-1)
                        # dL/dw(l-1) = a(l-2)^T * dL/dz(l-1)

                        dL_dW = np.dot(zal[i].T, dL_dz)
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
                        # dL/dz(l-1) = f'(z(l-1)) dot dL/da(l-1)
                        dL_dz = layer.backward(zal[i]) * dL_dz

            # Predict
            pred_train = self.predict(X_train)
            pred_eval = self.predict(X_eval)

            # Evaluate
            train_acc = ut.mae(y_train, pred_train)
            val_acc = ut.mae(y_eval, pred_eval)
            
            # Save
            self.history.save(train_acc, val_acc)
            self.history.save_predict(X_eval, pred_eval, y_eval)
            self.history.save_model(self.layers, val_acc)

            print(f"Epoch {epoch+1}/{epochs} [", end="")

            progress_bar_length = 25
            progress = int((epoch/epochs)*progress_bar_length)
            for i in range(progress_bar_length):
                if(i<=progress):
                    print("=", end="")
                else:
                    print(".", end="")
            print("]")
            print(f"loss: {train_acc:.4f}, val_loss: {val_acc:.4f}")
            print("") 
        print(f"best-loss: {self.history.get_best_loss():.4f}")
        return self.history