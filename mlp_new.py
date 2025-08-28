import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from TensorT.tensor_scratch import TensorT
from ActivationFunctions import ActivationFunction

from WeightInitialization import WeightInitializer
from LossFunctions import LossFunction

class MLP_NEW:
    def __init__(self, input_size, hidden_layers, output_size,
                 weight_initialization='he_normal', activation_func='relu',
                 loss_function='binary_cross_entropy_loss', learning_rate=0.01):
        self.layer_sizes = [input_size] + hidden_layers + [output_size]
        self.num_layers = len(self.layer_sizes)
        self.learning_rate = learning_rate
        self.weight_init_method = weight_initialization
        self.activation_func_name = activation_func
        self.loss_func_name = loss_function
        
        # Use only the forward functions, backward handled by semantic operations
        self.hidden_activation, _ = ActivationFunction.get(self.activation_func_name)
        self.output_activation, _ = ActivationFunction.output_for_loss(self.loss_func_name)
        self.cost, _ = LossFunction.get(self.loss_func_name)
        self.initializer = WeightInitializer.get(self.weight_init_method)
        
        self.weights = []
        self.biases = []
        self._initialize_parameters()
        
        print("MLP initialized successfully.")
        print(f" - Architecture: {self.layer_sizes}")
        print(f" - Hidden Activation: {self.activation_func_name}")
        print(f" - Output Activation (from loss): {self.output_activation.__name__}")
        print(f" - Weight Initialization: {self.weight_init_method}")
        print(f" - Loss Function: {self.loss_func_name}")

    def _initialize_parameters(self):
        for i in range(1, self.num_layers):
            in_dim, out_dim = self.layer_sizes[i-1], self.layer_sizes[i]
            W = self.initializer(in_dim, out_dim)
            b_data = [[0.0] for _ in range(out_dim)]
            self.weights.append(TensorT(W))
            self.biases.append(TensorT(b_data))

    def forward(self, X):
        A = X
        for l in range(1, self.num_layers):
            W = self.weights[l-1]
            b = self.biases[l-1]
            Z = (W @ A) + b
            act = self.hidden_activation if l < self.num_layers - 1 else self.output_activation
            A = act(Z)
        return A

    def backward_and_update(self, X, Y):
        # Forward for graph
        AL = self.forward(X)
        # Compute loss node (semantic node)
        loss = self.cost(Y, AL)
        # Backprop (automatic, implemented in TensorT and backward_fn of loss)
        loss.backward_fn([[1.0]])
        # Update weights and biases
        for l in range(1, self.num_layers):
            # Each param's .grad (if implemented in your TensorT)
            W = self.weights[l-1]
            b = self.biases[l-1]
            if hasattr(W, "grad") and W.grad is not None:
                W.data = [[w - self.learning_rate * dw for w, dw in zip(ws, dws)]
                          for ws, dws in zip(W.data, W.grad)]
            if hasattr(b, "grad") and b.grad is not None:
                b.data = [[val - self.learning_rate * dval for val, dval in zip(bs, dbs)]
                          for bs, dbs in zip(b.data, b.grad)]

    def train(self, X, Y, epochs, print_cost_every=100):
        print(f"\n--- Starting Training for {epochs} epochs ---")
        start_time = time.time()
        for i in range(epochs):
            self.backward_and_update(X, Y)
            if i % print_cost_every == 0 or i == epochs - 1:
                AL = self.forward(X)
                cost = float(self.cost(Y, AL).data[0])
                print(f"Epoch {i:>4} | Cost: {cost:.6f}")
        dt = time.time() - start_time
        print(f"--- Training finished in {dt:.2f} seconds ---")

    def predict(self, X):
        AL = self.forward(X)
        # Binary case
        if self.loss_func_name == 'binary_cross_entropy_loss' and self.layer_sizes[-1] == 1:
            return [[1 if a[0] > 0.5 else 0] for a in AL.data]
        # Multiclass: argmax over axis 0
        if self.loss_func_name in ('categorical_cross_entropy_loss', 'cross_entropy_loss'):
            # Assume AL.data is shape (num_classes, num_samples)
            import numpy as np  # For argmax, unless you make a pure Python version
            data_t = list(zip(*AL.data))  # Transpose to (num_samples, num_classes)
            return [row.index(max(row)) for row in data_t]
        # Otherwise, raw output
        return AL.data
