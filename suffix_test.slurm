#! /bin/bash
#
#SBATCH --job-name=1089_BPI19_nap


# Build the environment
source .venv/bin/activate

# Cd $
cd /data/leuven/342/vsc34240

# Run the program
python start_experiments_s.py BPI19 false 1089