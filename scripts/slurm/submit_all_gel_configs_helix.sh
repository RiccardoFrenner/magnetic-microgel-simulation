#!/bin/bash

module load compiler/gnu/14.1 mpi/openmpi/4.1
CLUSTER_FFTW3_VERSION=3.3.10
CLUSTER_BOOST_VERSION=1.82.0
CLUSTER_PYTHON_VERSION=3.12.4
export BOOST_ROOT="${HOME}/bin/boost_mpi_${CLUSTER_BOOST_VERSION//./_}"
export Boost_DIR="${HOME}/bin/boost_mpi_${CLUSTER_BOOST_VERSION//./_}/lib/cmake/Boost-1.82.0"
export FFTW3_ROOT="${HOME}/bin/fftw_${CLUSTER_FFTW3_VERSION//./_}"
export LD_LIBRARY_PATH="${BOOST_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${FFTW3_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="${HOME}/espresso-4.2/build/src/python${PYTHONPATH:+:$PYTHONPATH}"
source "${HOME}/venv/bin/activate"


args=(
  --bond_type "fene"
  --steps_melt_eq_max 100000
  --n_chains 400
  --chain_length 80
  --n_agents_per_crosslink 2
  --continue_past_crosslinking 1
  --steps_gel_eq_max 10000000
  --initial_diff_steps 10000
  --visualize_mpl
)


# Array of parameters
# 3*3*2*2*3
run_ids=(0 1 3)
agent_ffs=("none" "harmonic" "exp")
crosslink_percentages=(0.2 0.7)
volume_fractions=(0.3 0.6)
bead_charges=(0.15)
# bead_charges=(0.05 0.15 0.25)

mkdir -p ${HOME}/data/mmgel/prodrun_0/
mkdir -p ${HOME}/logs/prodrun_0

# Loop over all combinations of parameters
for run_id in ${run_ids[@]}; do
  for agent_ff in ${agent_ffs[@]}; do
    for crosslink_percentage in ${crosslink_percentages[@]}; do
      for volume_fraction in ${volume_fractions[@]}; do
        for charge in ${bead_charges[@]}; do

        # Create a unique job name for this combination of parameters
        job_name="mmgel2025_run_${run_id}_ff_${agent_ff}_clp_${crosslink_percentage}_vf_${volume_fraction}_charge_${charge}"

        # Write a new SLURM batch script
        cat << EOF > $job_name.sh
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --time=24:00:00
#SBATCH --output=${HOME}/logs/prodrun_0/${job_name}.out
#SBATCH --error=${HOME}/logs/prodrun_0/${job_name}.err
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --partition=cpu-single


mpiexec --bind-to core --map-by core ${HOME}/espresso-4.2/build/pypresso src/main.py ${HOME}/data/mmgel/prodrun_0/${job_name} --agent_force_field "$agent_ff" --crosslink_percentage $crosslink_percentage --volume_fraction $volume_fraction --charge_per_bead $charge ${args[@]}
EOF

          # Submit the job
          sbatch $job_name.sh
        done
      done
    done
  done
done
