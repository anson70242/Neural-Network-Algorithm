import numpy as np

class SOM():
    def __init__(self, input_dim: int, o_row: int, o_column: int):
        self.input_dim = input_dim
        self.o_row = o_row
        self.o_column = o_column
        
        # Init weight
        self.weights = np.random.randn(o_row, o_column, input_dim)

    @staticmethod
    def squared_e_distance(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        if x.shape != y.shape:
            raise ValueError("Input vectors must have the same shape")
        return np.sum((x - y)**2)
    
    # Finding Best Matching Unit(BMU)
    # Shortest distance
    def find_bmu(self, input_vector):
        input_vec = np.asarray(input_vector)
        min_dist_sq = np.inf # minimum distance squared
        bmu_location = (0, 0)
        
        for r in range(self.o_row):
            for c in range(self.o_column):
                weight_vector = self.weights[r, c, :]
                dist_sq = SOM.squared_e_distance(input_vec, weight_vector)

                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    bmu_location = (r, c)

        return bmu_location
    
    @staticmethod
    def find_neighbor(bmu_loc, neuron_loc, R):
        """
        R: the radius of the neighborhood
        """
        # 1. Calculate the squared Euclidean distance on the grid
        dist_sq = SOM.squared_e_distance(neuron_loc, bmu_loc)

        # 2. Calculate the influence h using the Gaussian neighborhood function
        #    Add a check for R being too small to avoid numerical issues
        if R < 1e-5:  # If R approaches 0
             # Only the BMU itself (distance is 0) has an influence of 1.0, others have 0.0
            return 1.0 if dist_sq == 0 else 0.0
        else:
            # exp(-distance_sq / R) with dist
            # exp(-distance_sq / (2 * R^2)) # Gaussian Function with dist_sq
            return np.exp(-dist_sq / (2 * R**2))

    def update_weight(self, input_vector, bmu_location, lr: float, R: float):
        """
        Updates the weights of all neurons in the SOM grid.

        The update rule is:
        W_new(r, c) = W_old(r, c) + lr * h(BMU, (r,c), R) * (input_vector - W_old(r, c))

        where:
        - W(r, c) is the weight vector of the neuron at grid location (r, c).
        - lr is the learning rate.
        - h is the neighborhood function value (influence).
        - R is the neighborhood radius.
        - BMU is the location of the Best Matching Unit.
        - input_vector is the current input data sample.

        Args:
            input_vector: The input data sample (should match input_dim).
            bmu_location: Tuple (row, col) representing the location of the Best Matching Unit.
            lr: The learning rate (eta) for the current iteration.
            R: The neighborhood radius (sigma) for the current iteration.
        """
        input_vec = np.asarray(input_vector)

        # Iterate through all neurons in the grid
        for r in range(self.o_row):
            for c in range(self.o_column):
                neuron_loc = (r, c)
                # Get the current weight vector of the neuron
                weight_vector = self.weights[r, c, :]

                # Calculate the neighborhood influence 'h' for this neuron
                influence = SOM.find_neighbor(bmu_location, neuron_loc, R)

                # Calculate the weight update delta
                # delta_w = learning_rate * influence * (input_vector - current_weight_vector)
                delta_w = lr * influence * (input_vec - weight_vector)

                # Update the neuron's weight vector
                self.weights[r, c, :] += delta_w