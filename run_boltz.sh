#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=run_boltz
#SBATCH --output=/home/ethanz/slurm/slurm-%j.out

source /home/ethanz/boltz-likelihoods/.venv/bin/activate

# Args applied to all types of inference.
MAIN_ARGS=(
    --model boltz1 \
    --max_parallel_samples 250 \
    --out_dir ./predictions/ \
    --use_msa_server \
    --confidence False \
    --mode langevin \
    --experiment_name chignolin_multistate
    --head_init /home/ethanz/boltz-likelihoods/conditioning/chignolin
    --slurm_path /home/ethanz/boltz-likelihoods/run_boltz.sh
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
LANGEVIN_ARGS=(
    --langevin_sampling_steps 250000 \
    --langevin_eps 0.00001 \
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
    examples/chignolin.yaml \
    "${LANGEVIN_ARGS[@]}" \
    "${MAIN_ARGS[@]}" \
    "${PRED_ARGS[@]}"
   

