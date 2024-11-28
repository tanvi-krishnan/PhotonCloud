import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse

# Load data from the input file
# This contains the photon scattering events, stored as a structured object.
data = np.load("photon_data.npy", allow_pickle=True)

# Set global font size for all elements in matplotlib plots
plt.rcParams.update({'font.size': 18})  # Adjust font size to make plots more readable.

def save_plot(fig, filename):
    """
    Saves the given figure to a file and closes it.

    Parameters:
    - fig: The matplotlib figure object to save.
    - filename: The name of the file to save the figure to.
    """
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def plot_scattering_histogram(times, delta_ts, mu_a, max_time, num_photons):
    """
    Plots a histogram of the scattering events as a function of time.

    Parameters:
    - times: List of scattering event times.
    - delta_ts: List of time intervals for scattering events.
    - mu_a: Absorption coefficient, used for the analytical solution.
    - max_time: Maximum simulation time.
    - num_photons: Total number of photons simulated.
    """
    fig = plt.figure(figsize=(10, 6))
    plt.hist(times, density=True, bins=50, color='skyblue', edgecolor='black', alpha=0.7)  # Histogram of event times
    plt.xlabel('Time')
    plt.ylabel('Weighted Scattering Events')
    plt.title('Histogram of Scattering Events in Time', fontsize=20)

    # Overlay analytical solution as a red line
    time_bins = np.linspace(0, max_time, 200)
    expected_counts = num_photons * np.exp(-mu_a * time_bins)
    expected_counts /= np.trapz(expected_counts, time_bins)  # Normalize the curve
    plt.plot(time_bins, expected_counts, 'r-', label='Analytical Solution $N(t) = N_0 e^{-\\mu_a t}$')

    plt.legend()
    plt.grid(True)
    save_plot(fig, "scattering_events_histogram.png")  # Save the plot as a file

def get_positions_at_time(data, target_time):
    """
    Extracts photon positions closest to a target time.

    Parameters:
    - data: Photon scattering data (positions and times).
    - target_time: The target time to analyze.

    Returns:
    - positions: Array of positions of photons at the target time.
    """
    positions = []
    for photon_data in data:
        for pos, event_time, delta_t in photon_data:
            if np.abs(event_time - target_time) <= 0.05:  # Check if event is within ±0.05 of target time
                positions.append(pos)
                break  # Only take the first valid event for each photon
    return np.array(positions)

def calculate_distances(positions):
    """
    Calculates the Euclidean distance of each position from the origin.

    Parameters:
    - positions: Array of 3D positions.

    Returns:
    - distances: Array of radial distances.
    """
    return np.linalg.norm(positions, axis=1)

def compute_delta_x_and_delta_t(data):
    """
    Computes the average squared step size and time interval from photon data.

    Parameters:
    - data: Photon scattering data (positions and times).

    Returns:
    - delta_x_squared: Average squared step size.
    - delta_t: Average time interval.
    """
    delta_x_squared_list = []
    delta_t_list = []

    for photon_events in data:
        positions = [event[0] for event in photon_events]
        times = [event[1] for event in photon_events]

        # Compute squared step distances and time intervals for the current photon
        step_distances_squared = [
            (torch.norm(torch.tensor(positions[i]) - torch.tensor(positions[i - 1]))**2).item()
            for i in range(1, len(positions))
        ]
        time_intervals = [times[i] - times[i - 1] for i in range(1, len(times))]

        # Add to the averages if there are valid data points
        if step_distances_squared:
            delta_x_squared_list.append(np.mean(step_distances_squared))
        if time_intervals:
            delta_t_list.append(np.mean(time_intervals))

    # Compute overall averages
    delta_x_squared = np.mean(delta_x_squared_list) if delta_x_squared_list else 0
    delta_t = np.mean(delta_t_list) if delta_t_list else 0
    return delta_x_squared, delta_t

def theoretical_gaussian_distribution(r_vals, D, target_time):
    """
    Computes the theoretical Gaussian distribution for radial distances in 3D diffusion.

    Parameters:
    - r_vals: Array of radial distances.
    - D: Diffusion coefficient.
    - target_time: The time at which to calculate the distribution.

    Returns:
    - gaussian: Theoretical Gaussian distribution.
    """
    gaussian = (r_vals**2 / (4 * np.pi * D * target_time)**(3/2)) * np.exp(-r_vals**2 / (4 * D * target_time))
    gaussian /= np.trapz(gaussian, r_vals)  # Normalize the Gaussian
    return gaussian

# Main execution block
if __name__ == "__main__":
    # Argument parser to accept user inputs
    parser = argparse.ArgumentParser(description="Photon scattering simulation analysis")
    parser.add_argument("--mu_a", type=float, required=True, help="Absorption coefficient")
    parser.add_argument("--target_time", type=float, required=True, help="Target time for position analysis")
    
    args = parser.parse_args()

    # Parameters
    max_time = 20.0  # Maximum simulation time
    num_photons = len(data)  # Total number of photons in the dataset

    # Collect scattering event times and time intervals
    times = []
    delta_ts = []
    for photon_events in data:
        for event in photon_events:
            position, t, delta_t = event
            times.append(t)
            delta_ts.append(delta_t)

    # Plot histogram of scattering events
    plot_scattering_histogram(times, delta_ts, args.mu_a, max_time, num_photons)

    # Get photon positions at the target time
    positions_at_target_time = get_positions_at_time(data, args.target_time)

    # Compute Δx^2 and Δt
    delta_x_squared, delta_t = compute_delta_x_and_delta_t(data)
    print("Δx^2 (Average Squared Step Size):", delta_x_squared)
    print("Δt (Average Time Interval):", delta_t)

    # Compute diffusion coefficient in 3D
    D = delta_x_squared / (6 * delta_t)  # Diffusion coefficient in 3D
    print("Diffusion Coefficient (D):", D)

    # Compute radial distances and theoretical Gaussian distribution
    r_positions = calculate_distances(positions_at_target_time)
    r_vals = np.linspace(r_positions.min(), r_positions.max(), 100)
    theoretical_gaussian = theoretical_gaussian_distribution(r_vals, D, args.target_time)

    # Plot radial distribution histogram with theoretical Gaussian
    fig2 = plt.figure(figsize=(10, 6))
    plt.hist(r_positions, density=True, bins=50, alpha=0.6, color='blue', label="Simulated Data")
    plt.plot(r_vals, theoretical_gaussian, 'r-', label="Theoretical Gaussian")
    plt.title(f"Photon Radial Distribution at Time t = {args.target_time}")
    plt.xlabel("Radial Distance (r)")
    plt.ylabel("Density")
    plt.legend()
    save_plot(fig2, "radial_distribution_sim.png")  # Save the plot as a file

    # Summary of results
    print("Plots saved as 'scattering_events_histogram.png' and 'radial_distribution_sim.png'")
