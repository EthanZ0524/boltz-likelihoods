#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=run_boltz

source .venv/bin/activate

# Set the path to this file to save it to every run's outdir. 
SCRIPT_PATH=run_boltz.sh

# The following arguments are relevant to every run, regardless of mode.
# --------------------------------------------------------------------------- #
# Change the four arguments below as needed.
YAML=examples/chignolin.yaml
MODE=umbrella
EXP_NAME=chignolin_umbrella_test
# HEAD_INIT=conditioning/trpcage # Comment this line out if not providing head_init.


MAIN_ARGS=(
    --model boltz1 \
    --max_parallel_samples 20 \
    --out_dir ./predictions/ \
    --use_msa_server \
    --confidence False \
    --output_format pdb \
    --save_conditioning_args False \
    --mode "$MODE" \
    --experiment_name "$EXP_NAME" \
    --slurm_path "$SCRIPT_PATH"
    --accelerator cpu
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

# Score args, used by both Langevin and umbrella sampling to set the
# score's time value.
SCORE_ARGS=(
    --diffusion_stop 180
)


# The following arguments pertain to specific inference modes.
# A mode's arguments do not affect runs of other modes. 
# --------------------------------------------------------------------------- #

# Args for likelihood calculation.
LIKELIHOOD_ARGS=(
    --likelihood_mode hutchinson \
    --hutchinson_samples 20 \
    --ode_batch_size 10 \
)

# Args for Langevin sampling.
LANGEVIN_ARGS=(
    --langevin_sampling_steps 250000 \
    --langevin_eps 0.00001 \
    --langevin_noise_scale 1.0 \
    --replicates 5
)

# Args for umbrella sampling.
UMBRELLA_ARGS=(
    --umbrella_functor Chignolin \
    --umbrella_json \
    --umbrella_steps 100
)

# Main run function.
boltz predict \
    "$YAML" \
    "${LANGEVIN_ARGS[@]}" \
    "${ODE_ARGS[@]}" \
    "${MAIN_ARGS[@]}" \
    "${PRED_ARGS[@]}" \
    "${LIKELIHOOD_ARGS[@]}" \
    "${SCORE_ARGS[@]}" \
    "${UMBRELLA_ARGS[@]}"

