#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=run_boltz
#SBATCH --output=/home/dunne/boltz-likelihoods/slurm/slurm-%j.out
#SBATCH --error=/home/dunne/boltz-likelihoods/slurm/slurm-%j.err

source .venv/bin/activate

# Set the path to this file to save it to every run's outdir. 
SCRIPT_PATH=run_boltz_likelihood.sh

# The following arguments are relevant to every run, regardless of mode.
# --------------------------------------------------------------------------- #
# Change the four arguments below as needed.
YAML=/home/dunne/boltz-likelihoods/examples/chignolin.yaml
MODE=likelihood  # Options: likelihood, langevin, predict_diff, predict_pfode
EXP_NAME=chignolin_ll
HEAD_INIT=/home/dunne/boltz-likelihoods/boltz_results_chignolin_pred_diff/predictions/chignolin/  # Comment this line out if not providing head_init.


MAIN_ARGS=(
    --model boltz1 \
    --max_parallel_samples 5 \
    --out_dir ./ \
    --use_msa_server \
    --confidence False \
    --output_format pdb \
    --save_conditioning_args False \
    --mode "$MODE" \
    --experiment_name "$EXP_NAME" \
    --slurm_path "$SCRIPT_PATH"
)

if [[ -n "$HEAD_INIT" ]]; then
  MAIN_ARGS+=(--head_init "$HEAD_INIT")
fi

# The following arguments are relevant to multiple modes.
# --------------------------------------------------------------------------- #

# ODE parameters, used by both likelihood calcs and ODE deterministic sampling.
ODE_ARGS=(
    --atol 0.000001 \
    --rtol 0.001
)

# 'Structure prediction' args, used by both Langevin and structure 
# prediction rollouts.
# step_scale affects only diffusion sampling, whereas diffusion_samples also 
# pertains to ODE sampling.
PRED_ARGS=(
    --step_scale 1.0 \
    --diffusion_samples 1 \
)


# The following arguments pertain to specific inference modes.
# A mode's arguments do not affect runs of other modes. 
# --------------------------------------------------------------------------- #

# Args for likelihood calculation.
LIKELIHOOD_ARGS=(
    --likelihood_mode jac \
    --hutchinson_samples 1
)

# Args for Langevin sampling.
LANGEVIN_ARGS=(
    --langevin_sampling_steps 5000 \
    --langevin_eps 0.001 \
    --langevin_noise_scale 1.0 \
    --diffusion_stop 195 \
    --replicates 1
)

# Main run function.
boltz predict \
    "$YAML" \
    "${LANGEVIN_ARGS[@]}" \
    "${ODE_ARGS[@]}" \
    "${MAIN_ARGS[@]}" \
    "${PRED_ARGS[@]}" \
    "${LIKELIHOOD_ARGS[@]}"
   

