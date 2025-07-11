#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=run_boltz

source /home/ethanz/boltz-likelihoods/.venv/bin/activate
boltz predict examples/aa.yaml \
    --diffusion_samples 5 \
    --model boltz1 \
    --step_scale 0.75 \ 
    # Low step_scale corresponds to higher sample diversity.
    --use_msa_server \
    --confidence False \
    --langevin True \
    --langevin_sampling_steps 500 \
    --langevin_eps 0.000001 \
    --langevin_noise_scale 1.0 \
    --diffusion_stop 195 \
    --output_format pdb \
    --experiment_name aa_test