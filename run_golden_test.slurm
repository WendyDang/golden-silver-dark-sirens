#!/bin/bash
#SBATCH --job-name=golden_h0
#SBATCH --output=logs/golden_test_%j.out
#SBATCH --error=logs/golden_test_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /hildafs/projects/phy220048p/dangy/gwcosmo

cd /hildafs/projects/phy220048p/share/gwbench_dark_siren/golden-silver-dark-sirens

echo "Starting golden siren H0 analysis"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Start time : $(date)"

python SDS_bilby.py

echo "End time   : $(date)"
