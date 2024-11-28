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
parser.add_argument("--num_flows", type=int, required=True, help="Number of flow layers")
parser.add_argument("--num_epochs", type=int, required=True, help="Number of training epochs")
parser.add_argument("--num_samples", type=int, required=True, help="Number of samples for final distribution")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--target_time", type=float, required=True, help="Target time for sampling analysis")
parser.add_argument("--max_time", type=float, required=True, help="Max time for time distribution")
args = parser.parse_args()


# Set global font size for all elements in matplotlib
plt.rcParams.update({'font.size': 18})  # Adjust font size as desired

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
        # Apply mask to separate active and passive parts for all dimensions
        mask = self.mask.view(1, -1).expand(x.shape[0], -1)  # Expand mask to match the batch size and input dimensions
        x_mask = x * mask  # Masked part remains fixed

        # Use x_mask (passive part) to generate s and t for active parts
        s = self.s_func(x_mask) * self.scale
        s = torch.clamp(s, min=-5.0, max=5.0)
        t = self.t_func(x_mask)

        # Apply transformation to active parts
        y = x_mask + (1 - mask) * (x * torch.exp(s) + t)
        log_det_jac = ((1 - mask) * s).sum(-1)
        return y, log_det_jac

    def inverse(self, y):
        # Apply mask to separate active and passive parts for all dimensions
        mask = self.mask.view(1, -1).expand(y.shape[0], -1)  # Expand mask to match the batch size and input dimensions
        y_mask = y * mask  # Masked part remains fixed

        # Use y_mask (passive part) to generate s and t for active parts
        s = self.s_func(y_mask) * self.scale
        s = torch.clamp(s, min=-5.0, max=5.0)
        t = self.t_func(y_mask)

        # Apply inverse transformation to active parts
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

    def base_distribution(self):
        mean = torch.zeros(self.dim).to(device)
        cov_matrix = torch.eye(self.dim).to(device)
        return MultivariateNormal(mean, cov_matrix)

    def log_probability(self, x):
        log_prob = torch.zeros(x.shape[0], device=x.device)
        for layer in reversed(self.layers):
            x, inv_log_det_jac = layer.inverse(x)
            log_prob += inv_log_det_jac
        base_dist = self.base_distribution()
        log_prob += base_dist.log_prob(x)
        return log_prob

    def rsample(self, num_samples):
        base_dist = self.base_distribution()
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
            time_value = None  # Use the average time value for training
            optimizer.zero_grad()
            log_prob = flow_model.log_probability(batch_data)
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
    masks = create_alternating_masks(3 + 1, args.num_flows)  # 3 spatial dim +1 for time dimension
    flow_model = RealNVP(masks, hidden_size=32).to(device)
    optimizer = optim.Adam(flow_model.parameters(), lr=1e-4)

    # Train the model
    losses = train_flow_model(flow_model, dataset, optimizer, args.num_epochs, args.batch_size)

    # Plot and save the training loss
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("training_loss.png", bbox_inches='tight')
    print("Training completed and loss plot saved as 'training_loss.png'.")

    # Sampling and plotting final results
    with torch.no_grad():
        # Sample from the trained model
        samples, _ = flow_model.rsample(args.num_samples)
        samples = samples.cpu().numpy()

    # Filter samples based on target time
    time_column = samples[:, -1]  # Assuming the last column is time
    time_tolerance = 0.5
    filtered_samples = samples[(time_column >= args.target_time - time_tolerance) &
                               (time_column <= args.target_time + time_tolerance)]

    # Calculate radial distances for spatial dimensions only
    r_positions = np.linalg.norm(filtered_samples[:, :3], axis=1)
    r_vals = np.linspace(r_positions.min(), r_positions.max(), 100)

    # Estimate diffusion coefficient D
    # Note: this only generates valid theory curve for mu_a = 0 and g = 0
    delta_x = 0.5  # Obtain these values printed by test_plots.py
    delta_t = 0.5
    D = delta_x / (6 * delta_t)

    # Compute theoretical Gaussian distribution
    def theoretical_gaussian_distribution(r_vals, D, target_time):
        # 3D diffusion solution with r^2 factor
        gaussian = (r_vals**2 / (4 * np.pi * D * target_time)**(3/2)) * np.exp(-r_vals**2 / (4 * D * target_time))

        # Normalize the Gaussian to ensure it integrates to 1
        gaussian /= np.trapz(gaussian, r_vals)
        return gaussian

    theoretical_gaussian = theoretical_gaussian_distribution(r_vals, D, target_time=args.target_time)

    # Plot histogram and theoretical Gaussian for radial distances
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(r_positions, density=True, bins=50, alpha=0.6, color='blue', label="Simulated Data")
    ax.plot(r_vals, theoretical_gaussian, 'r-', label="Theoretical Gaussian")
    #ax.set_title(f"Photon Radial Distribution at t ≈ {args.target_time}")
    ax.set_xlabel("Radial Distance (r)")
    ax.set_ylabel("Density")
    ax.legend()
    plt.savefig("radial_distribution.png", bbox_inches='tight')
    print("Radial distribution plot saved as 'radial_distribution.png'.")

    # Plot marginalized distribution for time
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(time_column, density=True, bins=np.linspace(-1, args.max_time + 2, 51), alpha=0.6, color='green')
    #ax.set_title("Marginalized Time Distribution of Photon Positions")
    ax.set_xlabel("Time")
    ax.set_ylabel("Density")
    ax.set_xlim(-1, args.max_time+2)
    ax.legend()
    plt.savefig("marginalized_time_distribution.png", bbox_inches='tight')
    print("Marginalized time distribution plot saved as 'marginalized_time_distribution.png'.")

  
   # Filter samples based on specific times (0.1, 1, 10)
    time_targets = [1.0, 5.0, 10.0, 15.0]
    time_tolerance = 0.05  # You can adjust this tolerance to select a reasonable number of samples
  
    for target_time in time_targets:
        # Define grid boundaries based on the range of your positions
        x_min, x_max = -10, 10
        y_min, y_max = -10, 10

        # Create a grid of points over the specified range
        grid_size = 200  # Number of points along each dimension of the grid
        x = np.linspace(x_min, x_max, grid_size)
        y = np.linspace(y_min, y_max, grid_size)
        xx, yy = np.meshgrid(x, y)

        # Flatten the grid points and add the z and target time to each point
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        time_component = np.full((grid_points.shape[0], 1), target_time)  # Add the time component

        # Flatten the grid points and add the z and target time to each point
        
        # Marginalize over z by integrating log probabilities over sampled z values
        num_z_samples = 200  # Number of z samples for marginalization
        densities = np.zeros(grid_points.shape[0])
        for _ in range(num_z_samples):
            # Randomly sample z value within a reasonable range (e.g., between -5 and 5)
            z_sample = np.random.uniform(-10, 10, size=(grid_points.shape[0], 1))
            grid_points_with_z_and_time = np.hstack([grid_points, z_sample, time_component])

            # Convert the grid points to a PyTorch tensor
            grid_points_tensor = torch.tensor(grid_points_with_z_and_time, dtype=torch.float32).to(device)

            # Evaluate the log probability at each grid point using the trained flow model
            with torch.no_grad():
                log_probs = flow_model.log_probability(grid_points_tensor)
                densities += np.exp(log_probs.cpu().numpy())

        # Average to integrate over z
        densities /= num_z_samples
        # Average the log probabilities to marginalize over z
        
        # Convert log-probs to densities
                
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
        plt.clim(vmin=0, vmax=0.00005)
        # plt.title(f"Photon Position Continuous PDF at t ≈ {target_time}")
        plt.xlabel("x Position")
        plt.ylabel("y Position")
        plt.savefig(f"continuous_pdf_t_{target_time}.png", bbox_inches='tight')
        plt.close()

        print(f"Continuous heatmap of photon position PDF at t ≈ {target_time} saved as 'continuous_pdf_t_{target_time}.png'.")

        
