import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse

# Load data from the input file
data = np.load("photon_data.npy", allow_pickle=True)

# Set global font size for all elements in matplotlib
plt.rcParams.update({'font.size': 18})  # Adjust font size as desired

# Define a function to save plots instead of displaying them
def save_plot(fig, filename):
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)

# Define function to plot histogram of scattering events in time
def plot_scattering_histogram(times, delta_ts, mu_a, max_time, num_photons):
    fig = plt.figure(figsize=(10, 6))
    plt.hist(times, density=True, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Weighted Scattering Events')
    plt.title('Histogram of Scattering Events in Time', fontsize=20)

    # Analytical solution line
    time_bins = np.linspace(0, max_time, 200)
    expected_counts = num_photons * np.exp(-mu_a * time_bins)
    expected_counts /= np.trapz(expected_counts, time_bins)  # Normalize to integrate to 1
    plt.plot(time_bins, expected_counts, 'r-', label='Analytical Solution $N(t) = N_0 e^{-\\mu_a t}$')

    plt.legend()
    plt.grid(True)
    save_plot(fig, "scattering_events_histogram.png")

# Calculate positions at target_time
def get_positions_at_time(data, target_time):
    positions = []
    for photon_data in data:
        for pos, event_time, delta_t in photon_data:
            if np.abs(event_time - target_time) <= 0.05:
                positions.append(pos)
                break
    return np.array(positions)

# Calculate distances for radial distribution
def calculate_distances(positions):
    return np.linalg.norm(positions, axis=1)

# Calculate average step size and time interval
def compute_delta_x_and_delta_t(data):
    delta_x_squared_list = []
    delta_t_list = []

    for photon_events in data:
        positions = [event[0] for event in photon_events]
        times = [event[1] for event in photon_events]

        # Calculate squared step distances (Δx^2) and time intervals (Δt) for this photon
        step_distances_squared = [(torch.norm(torch.tensor(positions[i]) - torch.tensor(positions[i - 1]))**2).item() for i in range(1, len(positions))]
        time_intervals = [times[i] - times[i - 1] for i in range(1, len(times))]

        # Average squared step distance and time interval for this photon
        if step_distances_squared:
            delta_x_squared_list.append(np.mean(step_distances_squared))
        if time_intervals:
            delta_t_list.append(np.mean(time_intervals))

    # Average Δx^2 and Δt across all photons
    delta_x_squared = np.mean(delta_x_squared_list) if delta_x_squared_list else 0
    delta_t = np.mean(delta_t_list) if delta_t_list else 0
    return delta_x_squared, delta_t

# Define theoretical Gaussian for 2D or 3D
def theoretical_gaussian_distribution(r_vals, D, target_time):
    # 3D diffusion solution with r^2 factor
    gaussian = (r_vals**2 / (4 * np.pi * D * target_time)**(3/2)) * np.exp(-r_vals**2 / (4 * D * target_time))
    
    # Normalize the Gaussian to ensure it integrates to 1
    gaussian /= np.trapz(gaussian, r_vals)
    return gaussian

# Main execution
if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Photon scattering simulation analysis")
    parser.add_argument("--mu_a", type=float, required=True, help="Absorption coefficient")
    parser.add_argument("--target_time", type=float, required=True, help="Target time for position analysis")
    
    args = parser.parse_args()

    # Parameters
    max_time = 20.0
    num_photons = len(data)
    
    # Calculate histogram data for scattering events in time
    times = []
    delta_ts = []
    for photon_events in data:
        for event in photon_events:
            position, t, delta_t = event
            times.append(t)
            delta_ts.append(delta_t)

    plot_scattering_histogram(times, delta_ts, args.mu_a, max_time, num_photons)

    # Get positions at target time
    positions_at_target_time = get_positions_at_time(data, args.target_time)

    # Calculate Δx and Δt
    delta_x_squared, delta_t = compute_delta_x_and_delta_t(data)
    print("Δx^2 (Average of Squared Step Sizes):", delta_x_squared)
    print("Δt (Average Time Interval):", delta_t)

    # Calculate diffusion coefficient in 3D
    D = delta_x_squared / (6 * delta_t)  # Corrected diffusion coefficient in 3D
    print(D)
    # Calculate radial distances and theoretical Gaussian
    r_positions = calculate_distances(positions_at_target_time)
    r_vals = np.linspace(r_positions.min(), r_positions.max(), 100)
    theoretical_gaussian = theoretical_gaussian_distribution(r_vals, D, args.target_time)

    # Plot histogram and theoretical Gaussian for radial distances
    fig2 = plt.figure(figsize=(10, 6))
    plt.hist(r_positions, density=True, bins=50, alpha=0.6, color='blue', label="Simulated Data")
    plt.plot(r_vals, theoretical_gaussian, 'r-', label="Theoretical Gaussian")
    plt.title(f"Photon Radial Distribution at Time t = {args.target_time}")
    plt.xlabel("Radial Distance (r)")
    plt.ylabel("Density")
    plt.legend()
    save_plot(fig2, "radial_distribution_sim.png")

    # Final results summary
    print("Plots saved as 'scattering_events_histogram.png' and 'radial_distribution.png'")


