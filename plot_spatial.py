import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# Load data from the input file
data = np.load("photon_data.npy", allow_pickle=True)

# Set global font size for all elements in matplotlib
plt.rcParams.update({'font.size': 18})  # Adjust font size as desired

def plot_spatial_distribution(data, target_time, time_window=0.5, dim=3, azimuth=30, elevation=30):
    """
    Plot the spatial distribution of scattering events around a given target time, with adjustable view angles.

    Parameters:
    - data: List of scattering events for each photon. Each event is (position, time, delta_t).
    - target_time: The time around which to gather scattering events.
    - time_window: The range of time around target_time to include events.
    - dim: Dimension of the plot (2 for 2D, 3 for 3D).
    - azimuth: Azimuthal viewing angle for the 3D plot.
    - elevation: Elevation viewing angle for the 3D plot.
    """
    x_vals = []
    y_vals = []
    z_vals = []

    # Collect positions within the specified time window
    for photon_events in data:
        for event in photon_events:
            position, t, delta_t = event
            if abs(t - target_time) <= time_window:
                x_vals.append(position[0].item())
                y_vals.append(position[1].item())
                if dim == 3:
                    z_vals.append(position[2].item())

    # Plot spatial distribution
    plt.figure(figsize=(9, 7.2))
    if dim == 3:
        ax = plt.axes(projection='3d')
        ax.scatter(x_vals, y_vals, z_vals, alpha=0.5, marker='o', color='blue')
        ax.view_init(elev=elevation, azim=azimuth)  # Set the view angle
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.set_title(f'Spatial Distribution of Scattering Events at t ≈ {target_time}')
    elif dim == 2:
        plt.scatter(x_vals, y_vals, alpha=0.5, marker='o', color='blue')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(f'Spatial Distribution of Scattering Events at t ≈ {target_time}')
    else:
        raise ValueError("Dimension must be 2 or 3")

    plt.grid(True)
    plt.show()

# Main execution
if __name__ == "__main__":
    # Argument parser to take `dim` and `target_time` as inputs
    parser = argparse.ArgumentParser(description="Photon scattering spatial distribution plot")
    parser.add_argument("--target_time", type=float, required=True, help="Target time for plotting scattering events")
    parser.add_argument("--dim", type=int, choices=[2, 3], required=True, help="Dimension of the plot (2D or 3D)")
    
    args = parser.parse_args()

    # Static parameters for the plot
    time_window = 0.5  # Time window around the target time
    azimuth = 30  # Azimuth angle for 3D view
    elevation = 30  # Elevation angle for 3D view

    # Call the plot function with the specified target_time and dim
    plot_spatial_distribution(data, args.target_time, time_window, args.dim, azimuth, elevation)
