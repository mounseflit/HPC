#!/bin/bash

echo "Hello, Let's Climb Toubkal..."
ls -la
module load CUDA
module load GCC/12.3.0
module load imkl
make clean
nvcc deviceQuery.cu -o deviceQuery
sbatch exec.slurm
make gemm && ./gemm
sbatch exec.slurm
cat slurm.out
echo "You Reached the Peak of Toubkal, Congratulations!"
