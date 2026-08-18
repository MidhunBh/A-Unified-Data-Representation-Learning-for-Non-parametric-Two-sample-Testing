#!/bin/bash
#SBATCH --job-name=c2st_baseline_mnist
#SBATCH --output=mnist_baseline%j.out
# SBATCH --error=job_error_%j.txt
#SBATCH --time=1-00:00:00

# SBATCH --partition=gpu-a100

# SBATCH --partition=feit-gpu-a100
# SBATCH --qos=feit

#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn

#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

##Load anaconda
module load Anaconda3/2022.10

##Activate research environment
eval "$(conda shell.bash hook)"
conda activate research

##Run experiment
python3 c2st_mnist.py

echo -e "\n-----------------------------------------------"
echo -e "\nFinished with execution of c2st_mnist.py script\n"
echo -e "-----------------------------------------------\n"

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
