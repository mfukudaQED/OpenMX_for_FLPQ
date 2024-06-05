#!/bin/bash
#SBATCH -J test
#SBATCH -p i8cpu
#SBATCH -N 1
#SBATCH -t 0:30:0
#SBATCH --exclusive

cd ${SLURM_SUBMIT_DIR}

ulimit -s unlimited

module purge
#module load oneapi_compiler/2023.0.0 oneapi_mkl/2023.0.0 openmpi/4.1.5-oneapi-2023.0.0-classic
module load oneapi_compiler/2023.0.0 oneapi_mkl/2023.0.0 oneapi_mpi/2023.0.0

#make -j 32 >> make.log
#make -j 32 >> make.log
make > make.log
