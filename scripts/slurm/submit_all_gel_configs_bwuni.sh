#!/bin/bash


module load compiler/gnu/13.3 mpi/openmpi/4.1 devel/cmake/3.29.3 devel/cuda/12.0 lib/hdf5/1.14.4-gnu-13.3-openmpi-4.1 devel/python/3.12.3_gnu_13.3
CLUSTER_FFTW3_VERSION=3.3.10
CLUSTER_BOOST_VERSION=1.82.0
export BOOST_ROOT="${HOME}/bin/boost_mpi_${CLUSTER_BOOST_VERSION//./_}"
export FFTW3_ROOT="${HOME}/bin/fftw_${CLUSTER_FFTW3_VERSION//./_}"
export CUDA_ROOT="${HOME}/bin/cuda_12_0"
export LD_LIBRARY_PATH="${BOOST_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${FFTW3_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}${CUDA_HOME}/targets/x86_64-linux/lib/stubs"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}${CUDA_ROOT}/lib"
export PYTHONPATH="${HOME}/espresso-4.2/build/src/python${PYTHONPATH:+:$PYTHONPATH}"





# Array of parameters
run_ids=(0 1)
agent_ffs=("none" "harmonic" "exp")
meshwidths=(2.5 5.0)
charges_per_bead=(0.05 0.15 0.25)

mkdir -p ${HOME}/logs/production5

# Loop over all combinations of parameters
for run_id in ${run_ids[@]}; do
  for agent_ff in ${agent_ffs[@]}; do
    for meshwidth in ${meshwidths[@]}; do
      for charge_per_bead in ${charges_per_bead[@]}; do

        # Create a unique job name for this combination of parameters
        job_name="run_${run_id}_ff_${agent_ff}_meshw_${meshwidth}_charge_${charge_per_bead}"

        # Write a new SLURM batch script
        cat << EOF > $job_name.sh
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --time=60:00:00
#SBATCH --output=${HOME}/logs/production5/${job_name}.out
#SBATCH --error=${HOME}/logs/production5/${job_name}.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --partition=single


mpiexec --bind-to core --map-by core ${HOME}/software/espresso-4.2/build/pypresso src/main.py ${HOME}/data/mmgel/production5/${job_name} --agent_force_field "$agent_ff" --desired_mesh_width "$meshwidth" --charge_per_bead "$charge_per_bead"
EOF

        # Submit the job
        sbatch $job_name.sh

      done
    done
  done
done
