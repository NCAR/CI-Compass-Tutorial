#PBS -A CESM0029
#PBS -N job_jax_cpu
#PBS -q main
#PBS -j oe
#PBS -l walltime=00:05:00
#PBS -l select=1:ncpus=128:mpiprocs=128:mem=235G

SIZE=1024
STEPS=10000

module load conda
conda activate /glade/work/bdobbins/conda/ci_compass

export JAX_PLATFORMS=cpu
export JAX_CUDA_VISIBLE_DEVICES=""
python ${HOME}/ci_compass/src/jax/heat_diffusion_jax.py --size ${SIZE} --steps ${STEPS}

