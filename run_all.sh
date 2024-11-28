#!/bin/bash

# Define paths to the scripts
PHOTON_MC_SCRIPT="photon_MC.py"
PLOT_SPATIAL_SCRIPT="plot_spatial.py"
NF_SCRIPT="conditional_nf.py"
TEST_PLOTS_SCRIPT="test_plots.py"

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
    --num_photons 10000 \
    --mu_s 2 \
    --mu_a 0.2 \
    --g 0.7 \
    --max_time 20.0
check_success "photon_MC.py"

# Run plot_spatial.py with specified inputs
echo "Running test_plots.py..."
python $TEST_PLOTS_SCRIPT \
    --mu_a 0.2 \
    --target_time 10.0 
check_success "plot_spatial.py"

# Run plot_spatial.py with specified inputs
echo "Running plot_spatial.py..."
python $PLOT_SPATIAL_SCRIPT \
    --target_time 10.0 
check_success "plot_spatial.py"

# Run conditional_nf.py with specified inputs
echo "Running conditional_nf.py..."
python $NF_SCRIPT \
    --num_flows 10 \
    --num_epochs 8 \
    --num_samples 1000000 \
    --batch_size 64 \
    --target_time 10.0 \
    --max_time 20.0 
check_success "conditional_nf.py"

# Final message if all scripts ran successfully
echo "All scripts completed successfully!"
