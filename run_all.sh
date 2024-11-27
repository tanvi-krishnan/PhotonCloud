#!/bin/bash
#SBATCH -t 0-24:00
#SBATCH --mem=5000
#SBATCH --output=run_all_%j.out  # Corrected this line
#SBATCH --error=run_all_%j.err
#SBATCH --partition=arguelles_delgado
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Define paths to the scripts
PHOTON_MC_SCRIPT="photon_MC.py"
PLOT_SPATIAL_SCRIPT="plot_spatial.py"
NF_SCRIPT="conditional_nf.py"

# Function to check if the last command ran successfully
check_success() {
    if [ $? -ne 0 ]; then
        echo "Error: $1 failed to execute."
        exit 1
    fi
}

# Run photon_MC.py with specified inputs
echo "Running photon_MC.py..."
python $PHOTON_MC_SCRIPT \
    --dim 3 \
    --num_photons 10000 \
    --mu_s 2 \
    --mu_a 0 \
    --g 0 \
    --max_time 20.0
check_success "photon_MC.py"

# Run plot_spatial.py with specified inputs
echo "Running plot_spatial.py..."
python $PLOT_SPATIAL_SCRIPT \
    --target_time 10.0 \
    --dim 3
check_success "plot_spatial.py"

# Run conditional_nf.py with specified inputs
echo "Running conditional_nf.py..."
python $NF_SCRIPT \
    --dim 3 \
    --num_flows 6 \
    --num_epochs 5 \
    --num_samples 1000000 \
    --batch_size 64 \
    --target_time 10.0
check_success "conditional_nf.py"

# Final message if all scripts ran successfully
echo "All scripts completed successfully!"
