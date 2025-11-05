#! /bin/bash
#
#SBATCH --job-name=BPI12_nap


# Build the environment
source .venv/bin/activate

# Cd $
cd /data/leuven/342/vsc34240

# Run the program
python start_experiments.py BPI12 false