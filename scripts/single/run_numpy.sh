#PBS -A CESM0029
#PBS -N job_numpy
#PBS -q main
#PBS -j oe
#PBS -l walltime=00:05:00
#PBS -l select=1:ncpus=128:mpiprocs=128:mem=235G

SIZE=1024
STEPS=10000

module load conda
conda activate /glade/work/bdobbins/conda/ci_compass

mpirun python ${HOME}/ci_compass/src/numpy/heat_diffusion_mpi.py --size ${SIZE} --steps ${STEPS}

