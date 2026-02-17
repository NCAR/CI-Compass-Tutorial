#PBS -A CESM0029
#PBS -N job_jax_gpu
#PBS -q develop
#PBS -j oe
#PBS -l walltime=00:05:00
#PBS -l select=1:ncpus=64:mpiprocs=1:mem=235G:ngpus=1

SIZE=1024
STEPS=10000

module load conda
conda activate /glade/work/bdobbins/conda/ci_compass

export JAX_PLATFORMS=cuda
unset LD_LIBRARY_PATH
python ${HOME}/ci_compass/src/jax/heat_diffusion_jax.py --size ${SIZE} --steps ${STEPS}

