import torch
import numpy as np
import argparse

# Function to generate an isotropic initial direction in 3D
def generate_isotropic_direction():
    cos_theta = 1 - 2 * torch.rand(1)
    sin_theta = torch.sqrt(1 - cos_theta**2)
    phi = 2 * torch.pi * torch.rand(1)
    direction = torch.tensor([
        sin_theta * torch.cos(phi),
        sin_theta * torch.sin(phi),
        cos_theta
    ])

    return direction.flatten()

# Henyey-Greenstein scattering function
def henyey_greenstein_scattering(g, current_direction):
    current_direction = torch.tensor(current_direction, dtype=torch.float32)
    current_direction = current_direction / torch.norm(current_direction)

    if g != 0:
        cos_theta = (1 / (2 * g)) * (1 + g**2 - ((1 - g**2) / (1 - g + 2 * g * torch.rand(1))) ** 2)
    else:
        cos_theta = 1 - 2 * torch.rand(1)

    sin_theta = torch.sqrt(1 - cos_theta**2)
    phi = 2 * torch.pi * torch.rand(1)

    perpendicular_vector = torch.cross(current_direction, torch.tensor([1.0, 0.0, 0.0]))
    if torch.norm(perpendicular_vector) < 1e-6:
        perpendicular_vector = torch.cross(current_direction, torch.tensor([0.0, 1.0, 0.0]))
    perpendicular_vector = perpendicular_vector / torch.norm(perpendicular_vector)
    second_perpendicular = torch.cross(current_direction, perpendicular_vector)
    new_direction = (sin_theta * torch.cos(phi) * perpendicular_vector +
                        sin_theta * torch.sin(phi) * second_perpendicular +
                        cos_theta * current_direction)
    
    return new_direction

# Event-Based Method with Competing Scattering and Absorption
def photon_propagation_event_based_competing(num_photons, mu_s, mu_a, g, max_time=20.0):
    data = []  # To store the results
    mu_t = mu_s + mu_a  # Total interaction rate

    for photon in range(num_photons):
        pos = torch.zeros(3)  # Start at origin
        time_photon = 0.0
        event_data = []  # To store (position, time, delta_t) for each scattering event

        direction = generate_isotropic_direction()
        while time_photon < max_time:
            delta_t = -torch.log(torch.rand(1)) / mu_t
            time_photon += delta_t.item()

            if time_photon >= max_time:
                break

            pos = pos + direction * delta_t

            if torch.rand(1).item() < (mu_s / mu_t):
                event_data.append((pos.clone().numpy(), time_photon, delta_t.item()))
                direction = henyey_greenstein_scattering(g, direction).flatten()
            else:
                break

        data.append(event_data)

    return data

# Main execution
if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Photon propagation simulation with event-based method")
    parser.add_argument("--num_photons", type=int, default=10000, help="Number of photons to simulate")
    parser.add_argument("--mu_s", type=float, default=2.0, help="Scattering coefficient")
    parser.add_argument("--mu_a", type=float, default=0.0, help="Absorption coefficient")
    parser.add_argument("--g", type=float, default=0.0, help="Asymmetry parameter for scattering")
    parser.add_argument("--max_time", type=float, default=20.0, help="Maximum simulation time")
    
    args = parser.parse_args()

    # Run simulation
    data = photon_propagation_event_based_competing(
        num_photons=args.num_photons,
        mu_s=args.mu_s,
        mu_a=args.mu_a,
        g=args.g,
        max_time=args.max_time
    )

    # Save the data to an .npy file
    np.save("photon_data.npy", np.array(data, dtype=object))
    print("Data saved to photon_data.npy")
