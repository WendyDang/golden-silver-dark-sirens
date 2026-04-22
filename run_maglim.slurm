#!/bin/bash
#SBATCH -A phy220048p
#SBATCH -p RM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --job-name=maglim_study
#SBATCH --output=/hildafs/projects/phy220048p/share/gwbench_dark_siren/golden-silver-dark-sirens/maglim_study/slurm_%j.out
#SBATCH --error=/hildafs/projects/phy220048p/share/gwbench_dark_siren/golden-silver-dark-sirens/maglim_study/slurm_%j.err

# maglim_study/ must exist before submitting (sbatch opens log files before the
# script body runs). Create it with: mkdir -p maglim_study && sbatch run_maglim.slurm

set -euo pipefail

export PS1=""
unset PYTHONPATH
eval "$(conda shell.bash hook)"
conda activate /hildafs/projects/phy220048p/dangy/gwcosmo

cd /hildafs/projects/phy220048p/share/gwbench_dark_siren/golden-silver-dark-sirens

export PYTHONUNBUFFERED=1
python -u SDS_bilby_maglim.py

exit 0
