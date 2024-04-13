#!/bin/bash
#SBATCH --job-name=train_model              # create a short name for your job
#SBATCH --output=logs/%x-%j.log           	# Standard output in run directory, formatted: jobName-jobID.log
#SBATCH --error=logs/%x-%j-E.log           	# Standard error log
#SBATCH --account=eecs602w24_class         	# account (check with $ my_accounts)
#SBATCH --partition=spgpu                   # (Submit to partition: standard, gpu. viz, largemem, oncampus, debug)
#SBATCH --gres=gpu:1                      	# GPUs per node
#SBATCH --time=0-16:00:00                    # total run time limit (dd-hh:mm:ss)
#SBATCH --nodes=1                         	# node count
#SBATCH --tasks-per-node=1                	# total number of tasks across all nodes
#SBATCH --cpus-per-task=1                 	# cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=60gb                         	# total memory per node
#SBATCH --mail-type=END,FAIL              	# send mail if job fails
#SBATCH --export=ALL                      	# Copy environment

# Load modules
module purge
module load python/3.10.4
module load cuda/11.8.0
source /home/joshmah/spatial_int_map_diffuser/envs/spatial_int_maps_3_10_4/bin/activate

# run trainer
python train.py --config-path /home/joshmah/spatial_int_map_diffuser/spatial-intention-maps/config/experiments/ours/pushing_4-small_divider-ours.yml

# close modules
deactivate
module purge