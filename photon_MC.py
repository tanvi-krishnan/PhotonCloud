import torch
import numpy as np
import argparse

# Function to generate an isotropic initial direction in 3D
# This function creates a random unit vector in 3D space.
def generate_isotropic_direction():
    # Generate a random cos(theta) uniformly in [-1, 1]
    cos_theta = 1 - 2 * torch.rand(1)
    # Calculate sin(theta) using trigonometric identity
    sin_theta = torch.sqrt(1 - cos_theta**2)
    # Generate a random phi uniformly in [0, 2*pi]
    phi = 2 * torch.pi * torch.rand(1)
    # Calculate the direction vector using spherical coordinates
    direction = torch.tensor([
        sin_theta * torch.cos(phi),  # x-component
        sin_theta * torch.sin(phi),  # y-component
        cos_theta                   # z-component
    ])
    return direction.flatten()  # Return a flattened tensor representing the direction

# Henyey-Greenstein scattering function
# Models the scattering direction based on the asymmetry parameter g.
def henyey_greenstein_scattering(g, current_direction):
    # Normalize the current direction vector
    current_direction = torch.tensor(current_direction, dtype=torch.float32)
    current_direction = current_direction / torch.norm(current_direction)

    if g != 0:
        # Compute cos(theta) using the Henyey-Greenstein formula
        cos_theta = (1 / (2 * g)) * (1 + g**2 - ((1 - g**2) / (1 - g + 2 * g * torch.rand(1))) ** 2)
    else:
        # For isotropic scattering (g = 0), generate cos(theta) uniformly in [-1, 1]
        cos_theta = 1 - 2 * torch.rand(1)

    # Calculate sin(theta) and phi
    sin_theta = torch.sqrt(1 - cos_theta**2)
    phi = 2 * torch.pi * torch.rand(1)

    # Generate a perpendicular vector to the current direction
    perpendicular_vector = torch.cross(current_direction, torch.tensor([1.0, 0.0, 0.0]))
    # If the perpendicular vector is near zero, choose an alternative axis
    if torch.norm(perpendicular_vector) < 1e-6:
        perpendicular_vector = torch.cross(current_direction, torch.tensor([0.0, 1.0, 0.0]))
    perpendicular_vector = perpendicular_vector / torch.norm(perpendicular_vector)
    # Compute a second perpendicular vector
    second_perpendicular = torch.cross(current_direction, perpendicular_vector)

    # Combine components to compute the new scattering direction
    new_direction = (sin_theta * torch.cos(phi) * perpendicular_vector +
                     sin_theta * torch.sin(phi) * second_perpendicular +
                     cos_theta * current_direction)
    
    return new_direction

# Event-based simulation of photon propagation
# Tracks scattering and absorption events within a medium.
def photon_propagation_event_based_competing(num_photons, mu_s, mu_a, g, max_time=20.0):
    data = []  # List to store photon event data
    mu_t = mu_s + mu_a  # Total interaction rate (scattering + absorption)

    # Simulate each photon
    for photon in range(num_photons):
        pos = torch.zeros(3)  # Photon starts at the origin
        time_photon = 0.0     # Initialize photon time
        event_data = []       # List to store events for this photon

        # Generate initial isotropic direction
        direction = generate_isotropic_direction()
        while time_photon < max_time:
            # Calculate time to next interaction (exponentially distributed)
            delta_t = -torch.log(torch.rand(1)) / mu_t
            time_photon += delta_t.item()  # Update photon time

            # If the updated time exceeds max_time, exit loop
            if time_photon >= max_time:
                break

            # Update photon position based on the current direction
            pos = pos + direction * delta_t

            # Decide whether the event is scattering or absorption
            if torch.rand(1).item() < (mu_s / mu_t):
                # Event is scattering
                event_data.append((pos.clone().numpy(), time_photon, delta_t.item()))  # Store event data
                # Compute new scattering direction using Henyey-Greenstein function
                direction = henyey_greenstein_scattering(g, direction).flatten()
            else:
                # Event is absorption; photon is terminated
                break

        # Store all events for this photon
        data.append(event_data)

    return data

# Main block for command-line execution
if __name__ == "__main__":
    # Argument parser for configuring simulation parameters
    parser = argparse.ArgumentParser(description="Photon propagation simulation with event-based method")
    parser.add_argument("--num_photons", type=int, default=10000, help="Number of photons to simulate")
    parser.add_argument("--mu_s", type=float, default=2.0, help="Scattering coefficient")
    parser.add_argument("--mu_a", type=float, default=0.0, help="Absorption coefficient")
    parser.add_argument("--g", type=float, default=0.0, help="Asymmetry parameter for scattering")
    parser.add_argument("--max_time", type=float, default=20.0, help="Maximum simulation time")
    args = parser.parse_args()

    # Run photon propagation simulation
    data = photon_propagation_event_based_competing(
        num_photons=args.num_photons,  # Number of photons
        mu_s=args.mu_s,                # Scattering coefficient
        mu_a=args.mu_a,                # Absorption coefficient
        g=args.g,                      # Henyey-Greenstein asymmetry parameter
        max_time=args.max_time         # Maximum simulation time
    )

    # Save the simulation data to an .npy file
    np.save("photon_data.npy", np.array(data, dtype=object))
    print("Data saved to photon_data.npy")
