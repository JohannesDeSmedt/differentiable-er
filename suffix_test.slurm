#! /bin/bash
#
#SBATCH --job-name=BPI17_suffix

#SBATCH --cluster=“genius”
#SBATCH --ntasks=1 --cpus-per-task=20
$SBATCH —-account=“lp_lirisnlp”
#SBATCH --mem="64G"

# Build the environment
source .venv/bin/activate

Cd $
cd /scratch/leuven/342/vsc34240

# Run the program
python start_experiments.py BPI17 True 0.8