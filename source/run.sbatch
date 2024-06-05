#!/bin/bash
#SBATCH -J test
#SBATCH -p i8cpu
#SBATCH -N 4
#SBATCH -t 0:30:0
#SBATCH --exclusive

cd ${SLURM_SUBMIT_DIR}

export MKL_DEBUG_CPU_TYPE=5
export OMP_NUM_THREADS=4
unset I_MPI_PMI_LIBRARY

ulimit -s unlimited

module purge
module load intel_compiler/2020.2.254 intel_mkl/2020.2.254 intel_mpi/2020.2.254

# This specifies the options used to run srun. The "-N1 -n1" options are
# used to allocates a single core to each task.
srun="srun --exclusive --mem-per-cpu=1840 -N 1 -n 8 -c 4"

#parallel="parallel --delay 0.5 -j 16 --joblog runtask.log --resume"
#parallel="parallel --pipe -N 4 --round-robin -j 4 parallel --delay 0.1 -j 4 --joblog runtask.log --resume"


### for AB2 high-throughput band calc
exefilename=/home/i0017/i001700/program/openmx3.9/source_ABHTP/openmx
dirname=$PWD
#cddir="cd $dirname && mkdir {/} && cd $dirname/{/} && cp ../../flat_restart/{/}/{/}.md . && cp ../../flat_restart/{/}/{/}.dat# {/}.dat"
cddir="cd $dirname && mkdir {/} && cd $dirname/{/} && cp ../../datband/{/}.dat ."
fname="{/}"
task1="echo '$dirname/$fname' >>tmp"

parallel1="parallel --pipe -L 128 --round-robin -j 4 cat"
parallel2="parallel -a - -j 128 --joblog runtask.log --resume"

### succeeded
#cat filenamelist.dat | parallel --pipe -L 128 --round-robin -j 4 cat | parallel -a - -j 128 --joblog runtask.log --resume "$task1"
#cat filenamelist.dat | parallel --pipe -L 128 --round-robin -j 4 cat | parallel -a - -j 128 --joblog runtask.log --resume "echo  '$dirname/$fname' >>tmp"

cat filenamelist.dat | $parallel1 | $parallel2 "$task1" &
#cat filenamelist.dat | $parallel &
#cat dat/filenamelist.txt | $parallel "$cddir" &
#cat filenamelist.dat | $parallel "$cddir && $srun $exefilename {/}.dat  -nt  ${OMP_NUM_THREADS} > {/}.std" &
###srun --exclusive --mem-per-cpu=1840 -n 8 -c 4 -N 1 $exefilename H_H-case01.dat  -nt  ${OMP_NUM_THREADS} > H_H-case01.std &

wait




# This specifies the options used to run GNU parallel:
#
#   --delay of 0.2 prevents overloading the controlling node.
#
#   -j is the number of tasks run simultaneously.
#
#   The combination of --joblog and --resume create a task log that
#   can be used to monitor progress.
#
#parallel="parallel --delay 0.2 -j $SLURM_NNODES --joblog runtask.log --resume"

# Run a script, runtask.sh, using GNU parallel and srun. Parallel
# will run the runtask script for the numbers 1 through 128. To
# illustrate, the first job will run like this:
#
#   srun --exclusive -N1 -n1 ./runtask.sh arg1:1 > runtask.1
#
#$parallel "$srun ./runtask.sh arg1:{1} > runtask.sh.{1}" ::: {1..128}

# Note that if your program does not take any input, use the -n0 option to
# call the parallel command: 
#
#   $parallel -n0 "$srun ./run_noinput_task.sh > output.{1}" ::: {1..128}
