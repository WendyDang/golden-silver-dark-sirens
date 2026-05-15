#!/bin/bash
#SBATCH -A phy220048p
#SBATCH -p RM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --job-name=maglim_golden
#SBATCH --output=/hildafs/projects/phy220048p/share/gwbench_dark_siren/golden-silver-dark-sirens/maglim_study_golden/slurm_%j.out
#SBATCH --error=/hildafs/projects/phy220048p/share/gwbench_dark_siren/golden-silver-dark-sirens/maglim_study_golden/slurm_%j.err

# maglim_study_golden/ must exist before submitting.
# Create with: mkdir -p maglim_study_golden && sbatch run_maglim_golden.slurm

set -euo pipefail

export PS1=""
unset PYTHONPATH
eval "$(conda shell.bash hook)"
conda activate /hildafs/projects/phy220048p/dangy/gwcosmo

cd /hildafs/projects/phy220048p/share/gwbench_dark_siren/golden-silver-dark-sirens

export PYTHONUNBUFFERED=1
python -u SDS_bilby_maglim.py --siren-type golden --n-events 100

exit 0
