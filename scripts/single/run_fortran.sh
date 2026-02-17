#PBS -A CESM0029
#PBS -N job_fortran
#PBS -q main
#PBS -j oe
#PBS -l walltime=00:05:00
#PBS -l select=1:ncpus=128:mpiprocs=128:mem=235G

SIZE=1024
STEPS=10000

module load ncarcompilers
module load intel
module load cray-mpich

# Change to the Fortran src directory and make the executable, then return to where we were
pushd ${HOME}/ci_compass/src/fortran && make
popd

mpirun ${HOME}/ci_compass/src/fortran/heat_diffusion_mpi --size ${SIZE} --steps ${STEPS}

