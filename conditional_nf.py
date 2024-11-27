import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import seaborn as sns

# Set device to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Argument parser for command-line inputs
parser = argparse.ArgumentParser(description="Normalizing Flow for Photon Data Simulation")
parser.add_argument("--dim", type=int, required=True, help="Spatial dimension (e.g., 2 or 3)")
parser.add_argument("--num_flows", type=int, required=True, help="Number of flow layers")
parser.add_argument("--num_epochs", type=int, required=True, help="Number of training epochs")
parser.add_argument("--num_samples", type=int, required=True, help="Number of samples for final distribution")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--target_time", type=float, required=True, help="Target time for sampling analysis")
args = parser.parse_args()

# Define RealNVPNode with time treated as an input variable
class RealNVPNode(nn.Module):
    def __init__(self, mask, hidden_size):
        super(RealNVPNode, self).__init__()
        self.dim = len(mask)
        self.mask = nn.Parameter(mask, requires_grad=False)

        input_size = self.dim
        self.s_func = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, self.dim)
        )
        self.t_func = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, self.dim)
        )
        self.scale = nn.Parameter(torch.zeros(self.dim))

    def forward(self, x):
        mask = self.mask.view(1, -1).expand_as(x)  # Expand mask to match the batch size
        x_mask = x * mask
        s = self.s_func(x_mask) * self.scale
        s = torch.clamp(s, min=-5.0, max=5.0)
        t = self.t_func(x_mask)

        y = x_mask + (1 - mask) * (x * torch.exp(s) + t)
        log_det_jac = ((1 - mask) * s).sum(-1)
        return y, log_det_jac

    def inverse(self, y):
        mask = self.mask.view(1, -1).expand_as(y)  # Expand mask to match the batch size
        y_mask = y * mask
        s = self.s_func(y_mask) * self.scale
        s = torch.clamp(s, min=-5.0, max=5.0)
        t = self.t_func(y_mask)

        x = y_mask + (1 - mask) * (y - t) * torch.exp(-s)
        inv_log_det_jac = ((1 - mask) * -s).sum(-1)
        return x, inv_log_det_jac

# Stack RealNVP layers into a normalizing flow model
class RealNVP(nn.Module):
    def __init__(self, masks, hidden_size):
        super(RealNVP, self).__init__()
        self.dim = len(masks[0])
        self.hidden_size = hidden_size

        self.masks = nn.ParameterList([nn.Parameter(mask, requires_grad=False) for mask in masks])
        self.layers = nn.ModuleList([RealNVPNode(mask, self.hidden_size) for mask in self.masks])

    def base_distribution(self, time_value):
        # Create a time-dependent mean and covariance
        mean = torch.zeros(self.dim).to(device)
        cov_matrix = torch.diag(torch.ones(self.dim))
        return MultivariateNormal(mean, cov_matrix)

    def log_probability(self, x, time_value):
        log_prob = torch.zeros(x.shape[0], device=x.device)
        for layer in reversed(self.layers):
            x, inv_log_det_jac = layer.inverse(x)
            log_prob += inv_log_det_jac
        base_dist = self.base_distribution(time_value)
        log_prob += base_dist.log_prob(x)
        return log_prob

    def rsample(self, num_samples, time_value):
        base_dist = self.base_distribution(time_value)
        x = base_dist.sample((num_samples,))
        log_prob = base_dist.log_prob(x)
        for layer in self.layers:
            x, log_det_jac = layer.forward(x)
            log_prob += log_det_jac
        return x, log_prob

# Generate masks for RealNVP layers
def create_alternating_masks(dim, num_layers):
    masks = []
    for i in range(num_layers):
        mask = torch.zeros(dim)
        mask[i % dim::2] = 1
        masks.append(mask)
    return masks

# Define training loop
def train_flow_model(flow_model, data, optimizer, num_epochs, batch_size):
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            batch_data = batch[0].to(device)
            time_value = batch_data[:, -1].median().item()  # Use the average time value for training
            optimizer.zero_grad()
            log_prob = flow_model.log_probability(batch_data, time_value)
            loss = -log_prob.mean()
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(flow_model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.6f}")

    return losses

# Function to load and preprocess data from npy file
def load_and_preprocess_data(file_path):
    data = np.load(file_path, allow_pickle=True)
    flattened_data = []

    for photon in data:
        cumulative_time = 0
        for event in photon:
            position, t, dt = event
            cumulative_time += dt
            # Concatenate spatial position and cumulative time into a single sample
            sample = torch.cat([torch.tensor(position, dtype=torch.float32), torch.tensor([cumulative_time], dtype=torch.float32)])
            flattened_data.append(sample)

    # Convert the flattened data to a single PyTorch tensor
    all_features = torch.stack(flattened_data).to(device)  # Shape will be [num_events, spatial_dim + 1]
    return all_features

# Main block
if __name__ == '__main__':
    # Load data and prepare for training
    data = load_and_preprocess_data("photon_data.npy")
    dataset = TensorDataset(data)  # Wrap the data in a TensorDataset

    # Initialize the RealNVP model and optimizer
    masks = create_alternating_masks(args.dim + 1, args.num_flows)  # Include +1 for time dimension
    flow_model = RealNVP(masks, hidden_size=32).to(device)
    optimizer = optim.Adam(flow_model.parameters(), lr=1e-4)

    # Train the model
    losses = train_flow_model(flow_model, dataset, optimizer, args.num_epochs, args.batch_size)

    # Plot and save the training loss
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("training_loss.png")
    print("Training completed and loss plot saved as 'training_loss.png'.")

    # Sampling and plotting final results
    with torch.no_grad():
        # Sample from the trained model
        samples, _ = flow_model.rsample(args.num_samples, args.target_time)
        samples = samples.cpu().numpy()

    # Filter samples based on target time
    time_column = samples[:, -1]  # Assuming the last column is time
    time_tolerance = 0.5
    filtered_samples = samples[(time_column >= args.target_time - time_tolerance) &
                               (time_column <= args.target_time + time_tolerance)]

    # Calculate radial distances for spatial dimensions only
    r_positions = np.linalg.norm(filtered_samples[:, :args.dim], axis=1)
    r_vals = np.linspace(r_positions.min(), r_positions.max(), 100)

    # Estimate diffusion coefficient D
    delta_x = 0.5 # Example average distance; refine as needed
    delta_t = 0.5
    D = delta_x / (2*args.dim * delta_t)

    # Compute theoretical Gaussian distribution
    def theoretical_gaussian_distribution(r_vals, D, target_time, dim):
        if dim == 3:
            # 3D diffusion solution with r^2 factor
            gaussian = (r_vals**2 / (4 * np.pi * D * target_time)**(3/2)) * np.exp(-r_vals**2 / (4 * D * target_time))
        elif dim == 2:
            # 2D diffusion solution with r factor
            gaussian = (r_vals / (4 * np.pi * D * target_time)) * np.exp(-r_vals**2 / (4 * D * target_time))
        else:
            raise ValueError("Only 2D or 3D supported.")

        # Normalize the Gaussian to ensure it integrates to 1
        gaussian /= np.trapz(gaussian, r_vals)
        return gaussian

    theoretical_gaussian = theoretical_gaussian_distribution(r_vals, D, target_time=args.target_time, dim=args.dim)

    # Plot histogram and theoretical Gaussian for radial distances
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(r_positions, density=True, bins=50, alpha=0.6, color='blue', label="Simulated Data")
    ax.plot(r_vals, theoretical_gaussian, 'r-', label="Theoretical Gaussian")
    ax.set_title(f"Photon Radial Distribution at t ≈ {args.target_time}")
    ax.set_xlabel("Radial Distance (r)")
    ax.set_ylabel("Density")
    ax.legend()
    plt.savefig("radial_distribution.png")
    print("Radial distribution plot saved as 'radial_distribution.png'.")

    # Filter samples based on specific times (0.1, 1, 10)
    time_targets = [0.1, 1, 5, 10, 15]
    time_tolerance = 0.05  # You can adjust this tolerance to select a reasonable number of samples

    for target_time in time_targets:
        # Define grid boundaries based on the range of your positions
        x_min, x_max = -10, 10
        y_min, y_max = -10, 10

        # Create a grid of points over the specified range
        grid_size = 100  # Number of points along each dimension of the grid
        x = np.linspace(x_min, x_max, grid_size)
        y = np.linspace(y_min, y_max, grid_size)
        xx, yy = np.meshgrid(x, y)

        # Flatten the grid points and add the z and target time to each point
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])

        # Marginalize over z by averaging log probabilities over sampled z values
        num_z_samples = 100  # Number of z samples for marginalization
        log_probs = []

        for _ in range(num_z_samples):
            # Randomly sample z value within a reasonable range (e.g., between -5 and 5)
            z_sample = np.random.uniform(-5, 5, size=(grid_points.shape[0], 1))
            time_component = np.full((grid_points.shape[0], 1), target_time)  # Add the time component
            grid_points_with_z_and_time = np.hstack([grid_points, z_sample, time_component])

            # Convert the grid points to a PyTorch tensor
            grid_points_tensor = torch.tensor(grid_points_with_z_and_time, dtype=torch.float32).to(device)

            # Evaluate the log probability at each grid point using the trained flow model
            with torch.no_grad():
                log_prob = flow_model.log_probability(grid_points_tensor, target_time)
                log_probs.append(log_prob.cpu().numpy())

        # Average the log probabilities to marginalize over z
        log_probs = np.mean(log_probs, axis=0)
        densities = np.exp(log_probs)  # Convert log-probs to densities

        # Reshape densities to match the grid shape
        density_grid = densities.reshape(xx.shape)

        # Plot the continuous density solution as a heatmap
        plt.figure(figsize=(8, 6))
        plt.imshow(
            density_grid,
            extent=[x_min, x_max, y_min, y_max],
            origin='lower',
            cmap='inferno',  # Use a colormap that starts from black
            aspect='auto'
        )
        plt.colorbar(label='Density')
        plt.title(f"Photon Position Continuous PDF at t ≈ {target_time} (Marginalized over z)")
        plt.xlabel("x Position")
        plt.ylabel("y Position")
        plt.savefig(f"continuous_pdf_marginalized_t_{target_time}.png")
        plt.close()

        print(f"Continuous heatmap of photon position PDF at t ≈ {target_time} (marginalized over z) saved as 'continuous_pdf_marginalized_t_{target_time}.png'.")
