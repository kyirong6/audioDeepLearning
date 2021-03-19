import numpy as np

# save activations and derivatives
# implement backpropagation
# implement gradient descent
# implement train method
# train our network with dummy dataset
# make some predictions


class MLP:

    def __init__(self, num_inputs=3, num_hidden=[3, 5], num_outputs=2):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        # concatenate values into single list
        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]

        # initiate random weights
        self.weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            self.weights.append(w)

        # each array in activations are the activations for a given layer
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

        # each array in derivatives are the derivatives of weights
        # -1 because for a NN we only have layers - 1 weight matrices
        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i+1]))
            derivatives.append(d)
        self.derivatives = derivatives

    def forward_propagate(self, inputs):
        """Computes forward propagation of the network based on input signals.

        Args:
            inputs (ndarray): Input signals
        Returns:
            activations (ndarray): Output values
        """
        activations = inputs
        self.activations[0] = inputs

        for i, w in enumerate(self.weights):
            # calculate net inputs for a given layer
            net_inputs = np.dot(activations, w)

            # calculate the activations
            activations = self._sigmoid(net_inputs)
            # store activations, i+1 because the 0th index of activations are the inputs.
            self.activations[i+1] = activations

        return activations

    def back_propagate(self, error, verbose=False):
        # reversed because we need to go backwards
        # dE/dW_i = (y - a_[a+1]) s'(h_[i+1]) a_i
        # s'(h_[i+1]) = s(h_[i+1])(1 - s(h_[i+1])
        # s(h_[i+1]) = a_[i+1]
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            delta = error * self._sigmoid_derivative(activations)
            # ndarray([0.1, 0.2]) --> ndarray([[0.1, 0.2]]
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i]
            # ndarray([0.1, 0.2]) --> ndarray([[0.1], [0.2]]
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
            error = np.dot(delta, self.weights[i].T)

            if verbose:
                print(f"Derivatives for W{i}: {self.derivatives[i]}")
        return error


    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    # create an MLP
    mlp = MLP(2, [5], 1)

    # create some inputs
    input = np.array([0.1, 0.2])
    target = np.array([0.3])

    # perform forward prop
    output = mlp.forward_propagate(input)

    # calculate error
    error = target - output

    # back propagation
    mlp.back_propagate(error, verbose=True)

    # print results
