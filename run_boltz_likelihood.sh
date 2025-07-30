#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=run_boltz
#SBATCH --output=/home/dunne/boltz-likelihoods/slurm/slurm-%j.out
#SBATCH --error=/home/dunne/boltz-likelihoods/slurm/slurm-%j.err

source /home/dunne/boltz-likelihoods/.venv/bin/activate

# Args applied to all types of inference.
MAIN_ARGS=(
    --model boltz1 \
    --max_parallel_samples 250 \
    --out_dir ./predictions/ \
    --use_msa_server \
    --confidence False \
    --mode likelihood \
    --head_init /home/dunne/boltz-likelihoods/predictions/boltz_results_rfah_ecoli_samle/predictions/rfah_ecoli \
    --experiment_name rfah_ecol_likelihood \
    --slurm_path /home/dunne/boltz-likelihoods/run_boltz.sh
)

# ODE parameters, used by both likelihood calcs and ODE deterministic sampling.
ODE_ARGS=(
    --atol 0.00001
    --rtol 0.001
)

# All arguments below are mode-specific.
# A mode's arguments do not affect runs of other modes. 
# --------------------------------------------------------------------------- #

# Args for Langevin sampling.
# TODO: replicates_per_traj
LANGEVIN_ARGS=(
    --langevin_sampling_steps 250000 \
    --langevin_eps 0.000001 \
    --langevin_noise_scale 1.0 \
    --diffusion_stop 180
)

# Args for structure prediction.
PRED_ARGS=(
    --diffusion_samples 15 \
    --step_scale 0.75 \
    --save_conditioning_args False \
    --output_format pdb 
)

# Main run function.
boltz predict \
    examples/rfah_ecoli.yaml \
    "${ODE_ARGS[@]}" \
    "${LANGEVIN_ARGS[@]}" \
    "${MAIN_ARGS[@]}" \
    "${PRED_ARGS[@]}"
   

