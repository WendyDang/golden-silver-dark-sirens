#!/bin/bash
#SBATCH --job-name=golden_h0
#SBATCH -A phy220048p
#SBATCH -p RM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/hildafs/projects/phy220048p/share/gwbench_dark_siren/golden-silver-dark-sirens/logs/golden_test_%j.out
#SBATCH --error=/hildafs/projects/phy220048p/share/gwbench_dark_siren/golden-silver-dark-sirens/logs/golden_test_%j.err

set -euo pipefail

mkdir -p /hildafs/projects/phy220048p/share/gwbench_dark_siren/golden-silver-dark-sirens/logs

export PS1=""
unset PYTHONPATH
eval "$(conda shell.bash hook)"
conda activate /hildafs/projects/phy220048p/dangy/gwcosmo
export PYTHONUNBUFFERED=1

cd /hildafs/projects/phy220048p/share/gwbench_dark_siren/golden-silver-dark-sirens

echo "========================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Start time : $(date)"
echo "========================================"

python -u SDS_bilby.py

echo "========================================"
echo "End time   : $(date)"
echo "========================================"

exit 0
