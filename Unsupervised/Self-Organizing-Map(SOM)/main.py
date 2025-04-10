import sys
import os
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Determine project root directory to adjust Python path
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

# Import necessary custom modules
import helper
# print(f"--- Imported helper module from: {helper.__file__} ---") # For debugging import path

import model # If 'model' fails, try importing 'SOM' as model

# --- Function to Calculate and Save U-Matrix ---
def calculate_and_save_u_matrix(som_model, epoch, map_rows, map_cols, output_dir="Unsupervised/Self-Organizing-Map(SOM)/img"):
    """Calculates the U-Matrix for the current SOM state and saves it to a file."""
    print(f"\nCalculating U-Matrix for Epoch {epoch}...")
    u_matrix = np.zeros((map_rows, map_cols))
    # Define Euclidean distance using the squared distance method from SOM class
    euclidean_distance = lambda x, y: np.sqrt(som_model.squared_e_distance(x, y))

    # Iterate through each neuron in the grid
    for r in range(map_rows):
        for c in range(map_cols):
            current_weight = som_model.weights[r, c, :]
            total_distance = 0.0
            neighbor_count = 0
            # Check orthogonal neighbors (up, down, left, right)
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                # Check if neighbor is within grid boundaries
                if 0 <= nr < map_rows and 0 <= nc < map_cols:
                    neighbor_weight = som_model.weights[nr, nc, :]
                    total_distance += euclidean_distance(current_weight, neighbor_weight)
                    neighbor_count += 1

            # Calculate average distance to neighbors
            if neighbor_count > 0:
                u_matrix[r, c] = total_distance / neighbor_count
            # else: handle edge case of 1x1 map if necessary (shouldn't happen here)

    print("U-Matrix calculation finished.")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Define filename including epoch number
    filename = os.path.join(output_dir, f"u_matrix_epoch_{epoch}.png")

    # --- Plotting ---
    # Create a new figure for plotting to avoid overlap
    plt.figure(figsize=(8, 8))
    # Display the U-Matrix using a suitable colormap ('bone_r': dark=similar, light=dissimilar)
    plt.imshow(u_matrix, cmap='bone_r', interpolation='nearest')
    # Add title with epoch number
    plt.title(f'Unified Distance Matrix (U-Matrix) - Epoch {epoch}')
    # Add color bar for reference
    plt.colorbar(label='Average distance to neighbors')
    # Set ticks to match grid coordinates
    plt.xticks(np.arange(map_cols))
    plt.yticks(np.arange(map_rows))
    # Turn off grid lines for better U-Matrix visualization
    plt.grid(False)

    # Save the figure to the specified file
    plt.savefig(filename)
    # Close the figure to free memory and prevent it from being shown automatically later
    plt.close()
    print(f"Saved U-Matrix plot to: {filename}")

# --- Main Function ---
def main():
    # Load and prepare the Iris dataset
    dataset = helper.load_iris()
    X = dataset[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].to_numpy().tolist()
    y = dataset['Species'].to_numpy().tolist() # Original labels for stratify

    # Scale features and one-hot encode labels
    X_scaled = helper.minmax_scaler(X)
    y_one_hot = helper.one_hot_encode(y) # One-hot labels (kept for potential future use)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,        # Scaled features
        y_one_hot,       # One-hot encoded target (not directly used in unsupervised SOM training)
        test_size=0.3,   # 30% for test set
        train_size=0.7,  # 70% for train set
        random_state=42, # For reproducible splits
        shuffle=True,    # Shuffle data before splitting
        stratify=y       # Maintain class proportions using original labels 'y'
    )

    # --- SOM Training Parameters ---
    input_dim = len(X_train[0]) # Dimension of input vectors (number of features)
    map_rows = 8                # Height of the SOM grid
    map_cols = 8                # Width of the SOM grid
    epochs = 10000              # Number of training iterations (epochs) you set
    initial_learning_rate = 0.5 # Initial learning rate (eta_0)
    initial_radius = max(map_rows, map_cols) / 2.0 # Initial neighborhood radius (sigma_0)
    save_interval = 1000        # Set the interval for saving the U-Matrix (every N epochs)

    # Calculate time constant for learning rate and radius decay
    # Using epochs / log(initial_radius) is common for radius decay
    time_constant = epochs / np.log(initial_radius) if initial_radius > 1 else epochs

    # Print parameters
    print(f"Input dimension: {input_dim}")
    print(f"Map size: {map_rows}x{map_cols}")
    print(f"Epochs: {epochs}")
    print(f"Initial Learning Rate: {initial_learning_rate}")
    print(f"Initial Radius: {initial_radius:.2f}")
    print(f"Time Constant for decay: {time_constant:.2f}")
    print(f"Saving U-Matrix every {save_interval} epochs.")

    # --- Initialize the SOM ---
    som_model = model.SOM(input_dim=input_dim, o_row=map_rows, o_column=map_cols)
    print("SOM model initialized.")

    # --- Training Loop ---
    print("Starting SOM training...")
    num_samples = len(X_train)
    indices = np.arange(num_samples) # Indices for shuffling training data

    # Loop over epochs
    for epoch in range(epochs):
        # Shuffle training data for this epoch to avoid order bias
        np.random.shuffle(indices)
        X_train_shuffled = [X_train[i] for i in indices]

        # Calculate decaying learning rate (lr) and radius (R) for this epoch
        # Using exponential decay: value(t) = value0 * exp(-t / time_constant)
        current_lr = initial_learning_rate * np.exp(-epoch / time_constant)
        current_R = initial_radius * np.exp(-epoch / time_constant)

        # Optional: Prevent radius from collapsing too early (can be useful)
        # current_R = max(1.0, current_R) # Ensure radius is at least 1 grid unit

        # Iterate through each training sample in the shuffled list
        for i, input_vector in enumerate(X_train_shuffled):
            # 1. Find the Best Matching Unit (BMU) for the current input vector
            bmu_location = som_model.find_bmu(input_vector)

            # 2. Update weights of the BMU and its neighbors based on current lr and R
            som_model.update_weight(input_vector, bmu_location, current_lr, current_R)

        # --- Periodic U-Matrix Saving ---
        # Check if the current epoch (+1 because epoch starts from 0) is a multiple of save_interval
        if (epoch + 1) % save_interval == 0:
            calculate_and_save_u_matrix(som_model, epoch + 1, map_rows, map_cols)

        # Print progress (adjusted frequency for potentially long training)
        if (epoch + 1) % 100 == 0 or epoch == 0: # Print every 100 epochs
             print(f"Epoch {epoch + 1}/{epochs} completed - LR: {current_lr:.4f}, Radius: {current_R:.4f}")

    print("SOM training finished.")

    # --- Final U-Matrix Saving (Optional) ---
    # Save the U-Matrix for the final epoch if it wasn't saved exactly on the interval
    if epochs % save_interval != 0:
         calculate_and_save_u_matrix(som_model, epochs, map_rows, map_cols)

    # --- Evaluate Quantization Error (Training Set) ---
    # QE measures the average distance between input vectors and their BMU's weights
    # Here, we calculate the Mean Squared Quantization Error
    total_qe_train_sq = 0.0
    for input_vector in X_train:
        bmu_loc = som_model.find_bmu(input_vector)
        bmu_weight = som_model.weights[bmu_loc[0], bmu_loc[1], :]
        total_qe_train_sq += som_model.squared_e_distance(input_vector, bmu_weight)
    mean_qe_train_sq = total_qe_train_sq / num_samples
    print(f"\nMean Squared Quantization Error (Train): {mean_qe_train_sq:.4f}")

    # --- Evaluate Quantization Error (Test Set) ---
    # Here, we calculate the Mean Euclidean Quantization Error for comparison
    num_samples_test = len(X_test)
    if num_samples_test > 0:
        total_qe_test = 0.0
        for input_vector in X_test:
            bmu_loc = som_model.find_bmu(input_vector)
            bmu_weight = som_model.weights[bmu_loc[0], bmu_loc[1], :]
            total_qe_test += np.sqrt(som_model.squared_e_distance(input_vector, bmu_weight)) # Calculate Euclidean distance
        mean_qe_test = total_qe_test / num_samples_test
        print(f"Mean Euclidean Quantization Error (Test): {mean_qe_test:.4f}")
    else:
        print("No test data to evaluate.")

if __name__ == "__main__":
    main()