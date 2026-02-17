#PBS -A CESM0029
#PBS -N strong_numpy
#PBS -q main
#PBS -j oe
#PBS -l walltime=00:10:00
#PBS -l select=16:ncpus=128:mpiprocs=128:mem=235G

SIZE=1024
STEPS=10000

module load conda
conda activate /glade/work/bdobbins/conda/ci_compass

for nodes in 1 4 16; do    # Since we have square 2D domains of size^2, we're doing nodes^2 to keep the problem size the same
  CORES=$((nodes * 128))
  SQRT_NODES=$( echo "sqrt($nodes)" | bc)
  TOTAL_SIZE=${SIZE}
  echo "Running with ${nodes} nodes [ ${CORES} cores ] and a problem size of ${TOTAL_SIZE}^2"
  mpirun -n ${CORES} -ppn 128  python ${HOME}/ci_compass/src/numpy/heat_diffusion_mpi.py --size ${TOTAL_SIZE} --steps ${STEPS}
  echo ""
  
done


