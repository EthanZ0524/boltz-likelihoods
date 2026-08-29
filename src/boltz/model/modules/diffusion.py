# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang

from __future__ import annotations

from math import sqrt, ceil

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn import Module

import boltz.model.layers.initialize as init
from boltz.data import const
from boltz.model.loss.diffusion import (
    smooth_lddt_loss,
    weighted_rigid_align,
)
from boltz.model.modules.encoders import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    FourierEmbedding,
    PairwiseConditioning,
    SingleConditioning,
)
from boltz.model.modules.transformers import (
    ConditionedTransitionBlock,
    DiffusionTransformer,
)
from boltz.model.modules.utils import (
    LinearNoBias,
    compute_random_augmentation,
    center_random_augmentation,
    default,
    log,
)
from boltz.model.potentials.potentials import get_potentials

from boltz.lutils.cvs import CLASS_REGISTRY

from openmm.app import *
from openmm import *
import openmm.unit as unit
import mdtraj as md

from torchdiffeq import odeint
import torchode as to
from einops import rearrange, repeat, reduce, einsum
import h5py
import os
import json
import numpy as np

from tqdm import tqdm
import time

# Boltz diffusion's score network.
class DiffusionModule(Module):
    """Diffusion module"""

    def __init__(
        self,
        token_s: int,
        token_z: int,
        atom_s: int,
        atom_z: int,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        sigma_data: int = 16,
        dim_fourier: int = 256,
        atom_encoder_depth: int = 3,
        atom_encoder_heads: int = 4,
        token_transformer_depth: int = 24,
        token_transformer_heads: int = 8,
        atom_decoder_depth: int = 3,
        atom_decoder_heads: int = 4,
        atom_feature_dim: int = 128,
        conditioning_transition_layers: int = 2,
        activation_checkpointing: bool = False,
        offload_to_cpu: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the diffusion module.

        Parameters
        ----------
        token_s : int
            The single representation dimension.
        token_z : int
            The pair representation dimension.
        atom_s : int
            The atom single representation dimension.
        atom_z : int
            The atom pair representation dimension.
        atoms_per_window_queries : int, optional
            The number of atoms per window for queries, by default 32.
        atoms_per_window_keys : int, optional
            The number of atoms per window for keys, by default 128.
        sigma_data : int, optional
            The standard deviation of the data distribution, by default 16.
        dim_fourier : int, optional
            The dimension of the fourier embedding, by default 256.
        atom_encoder_depth : int, optional
            The depth of the atom encoder, by default 3.
        atom_encoder_heads : int, optional
            The number of heads in the atom encoder, by default 4.
        token_transformer_depth : int, optional
            The depth of the token transformer, by default 24.
        token_transformer_heads : int, optional
            The number of heads in the token transformer, by default 8.
        atom_decoder_depth : int, optional
            The depth of the atom decoder, by default 3.
        atom_decoder_heads : int, optional
            The number of heads in the atom decoder, by default 4.
        atom_feature_dim : int, optional
            The atom feature dimension, by default 128.
        conditioning_transition_layers : int, optional
            The number of transition layers for conditioning, by default 2.
        activation_checkpointing : bool, optional
            Whether to use activation checkpointing, by default False.
        offload_to_cpu : bool, optional
            Whether to offload the activations to CPU, by default False.

        """
        super().__init__()

        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys
        self.sigma_data = sigma_data

        self.single_conditioner = SingleConditioning(
            sigma_data=sigma_data,
            token_s=token_s,
            dim_fourier=dim_fourier,
            num_transitions=conditioning_transition_layers,
        )
        self.pairwise_conditioner = PairwiseConditioning(
            token_z=token_z,
            dim_token_rel_pos_feats=token_z,
            num_transitions=conditioning_transition_layers,
        )

        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            token_z=token_z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_feature_dim=atom_feature_dim,
            atom_encoder_depth=atom_encoder_depth,
            atom_encoder_heads=atom_encoder_heads,
            structure_prediction=True,
            activation_checkpointing=activation_checkpointing,
        )

        self.s_to_a_linear = nn.Sequential(
            nn.LayerNorm(2 * token_s), LinearNoBias(2 * token_s, 2 * token_s)
        )
        init.final_init_(self.s_to_a_linear[1].weight)

        self.token_transformer = DiffusionTransformer(
            dim=2 * token_s,
            dim_single_cond=2 * token_s,
            dim_pairwise=token_z,
            depth=token_transformer_depth,
            heads=token_transformer_heads,
            activation_checkpointing=activation_checkpointing,
            offload_to_cpu=offload_to_cpu,
        )

        self.a_norm = nn.LayerNorm(2 * token_s)

        self.atom_attention_decoder = AtomAttentionDecoder(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            atom_decoder_depth=atom_decoder_depth,
            atom_decoder_heads=atom_decoder_heads,
            activation_checkpointing=activation_checkpointing,
        )

    def forward(
        self,
        s_inputs,
        s_trunk,
        z_trunk,
        r_noisy,
        times,
        relative_position_encoding,
        feats,
        multiplicity=1,
        model_cache=None,
    ):
        """Computes the noise.

        Parameters
        ----------
        s_inputs : torch.tensor of shape ()
        s_trunk : torch.tensor of shape ()
        z_trunk : torch.tensor of shape (n_tokens, n_tokens, c_z)
        relative_position_encoding : torch.tensor of shape (n_tokens, n_tokens, c_z)
        
        Returns
        -------
        r_update : torch.tensor of shape ()

        """
        s, normed_fourier = self.single_conditioner(
            times=times,
            s_trunk=s_trunk.repeat_interleave(multiplicity, 0),
            s_inputs=s_inputs.repeat_interleave(multiplicity, 0),
        )

        if model_cache is None or len(model_cache) == 0:
            z = self.pairwise_conditioner(
                z_trunk=z_trunk, token_rel_pos_feats=relative_position_encoding
            )
        else:
            z = None

        # Compute Atom Attention Encoder and aggregation to coarse-grained tokens
        a, q_skip, c_skip, p_skip, to_keys = self.atom_attention_encoder(
            feats=feats,
            s_trunk=s_trunk,
            z=z,
            r=r_noisy,
            multiplicity=multiplicity,
            model_cache=model_cache,
        )

        # Full self-attention on token level
        a = a + self.s_to_a_linear(s)

        mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        a = self.token_transformer(
            a,
            mask=mask.float(),
            s=s,
            z=z,  # note z is not expanded with multiplicity until after bias is computed
            multiplicity=multiplicity,
            model_cache=model_cache,
        )
        a = self.a_norm(a)

        # Broadcast token activations to atoms and run Sequence-local Atom Attention
        r_update = self.atom_attention_decoder(
            a=a,
            q=q_skip,
            c=c_skip,
            p=p_skip,
            feats=feats,
            multiplicity=multiplicity,
            to_keys=to_keys,
            model_cache=model_cache,
        )

        return {"r_update": r_update, "token_a": a.detach()}


class OutTokenFeatUpdate(Module):
    """Output token feature update"""

    def __init__(
        self,
        sigma_data: float,
        token_s=384,
        dim_fourier=256,
    ):
        """Initialize the Output token feature update for confidence model.

        Parameters
        ----------
        sigma_data : float
            The standard deviation of the data distribution.
        token_s : int, optional
            The token dimension, by default 384.
        dim_fourier : int, optional
            The dimension of the fourier embedding, by default 256.

        """

        super().__init__()
        self.sigma_data = sigma_data

        self.norm_next = nn.LayerNorm(2 * token_s)
        self.fourier_embed = FourierEmbedding(dim_fourier)
        self.norm_fourier = nn.LayerNorm(dim_fourier)
        self.transition_block = ConditionedTransitionBlock(
            2 * token_s, 2 * token_s + dim_fourier
        )

    def forward(
        self,
        times,
        acc_a,
        next_a,
    ):
        next_a = self.norm_next(next_a)
        fourier_embed = self.fourier_embed(times)
        normed_fourier = (
            self.norm_fourier(fourier_embed)
            .unsqueeze(1)
            .expand(-1, next_a.shape[1], -1)
        )
        cond_a = torch.cat((acc_a, normed_fourier), dim=-1)

        acc_a = acc_a + self.transition_block(next_a, cond_a)

        return acc_a

# Boltz structure prediction diffusion module.
class AtomDiffusion(Module):
    """Atom diffusion module"""

    def __init__(
        self,
        score_model_args,
        num_sampling_steps=5,
        sigma_min=0.0004,
        sigma_max=160.0,
        sigma_data=16.0,
        rho=7,
        P_mean=-1.2,
        P_std=1.5,
        gamma_0=0.8,
        gamma_min=1.0,
        noise_scale=1.003,
        step_scale=1.5,
        coordinate_augmentation=True,
        compile_score=False,
        alignment_reverse_diff=False,
        synchronize_sigmas=False,
        use_inference_model_cache=False,
        accumulate_token_repr=False,
        **kwargs,
    ):
        """Initialize the atom diffusion module.

        Parameters
        ----------
        score_model_args : dict
            The arguments for the score model.
        num_sampling_steps : int, optional
            The number of sampling steps, by default 5.
        sigma_min : float, optional
            The minimum sigma value, by default 0.0004.
        sigma_max : float, optional
            The maximum sigma value, by default 160.0.
        sigma_data : float, optional
            The standard deviation of the data distribution, by default 16.0.
        rho : int, optional
            The rho value, by default 7.
        P_mean : float, optional
            The mean value of P, by default -1.2.
        P_std : float, optional
            The standard deviation of P, by default 1.5.
        gamma_0 : float, optional
            The gamma value, by default 0.8.
        gamma_min : float, optional
            The minimum gamma value, by default 1.0.
        noise_scale : float, optional
            The noise scale, by default 1.003.
        step_scale : float, optional
            The step scale, by default 1.5.
        coordinate_augmentation : bool, optional
            Whether to use coordinate augmentation, by default True.
        compile_score : bool, optional
            Whether to compile the score model, by default False.
        alignment_reverse_diff : bool, optional
            Whether to use alignment reverse diff, by default False.
        synchronize_sigmas : bool, optional
            Whether to synchronize the sigmas, by default False.
        use_inference_model_cache : bool, optional
            Whether to use the inference model cache, by default False.
        accumulate_token_repr : bool, optional
            Whether to accumulate the token representation, by default False.

        Defaults set in BoltzDiffusionParams.
        """
        super().__init__()
        self.score_model = DiffusionModule(
            **score_model_args,
        )
        if compile_score:
            self.score_model = torch.compile(
                self.score_model, dynamic=False, fullgraph=False
            )

        # parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.num_sampling_steps = num_sampling_steps
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.step_scale = step_scale
        self.coordinate_augmentation = coordinate_augmentation
        self.alignment_reverse_diff = alignment_reverse_diff
        self.synchronize_sigmas = synchronize_sigmas
        self.use_inference_model_cache = use_inference_model_cache

        self.accumulate_token_repr = accumulate_token_repr
        self.token_s = score_model_args["token_s"]
        if self.accumulate_token_repr:
            self.out_token_feat_update = OutTokenFeatUpdate(
                sigma_data=sigma_data,
                token_s=score_model_args["token_s"],
                dim_fourier=score_model_args["dim_fourier"],
            )

        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

    @property
    def device(self):
        return next(self.score_model.parameters()).device

    def c_skip(self, sigma):
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        return sigma * self.sigma_data / torch.sqrt(self.sigma_data**2 + sigma**2)

    def c_in(self, sigma):
        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma):
        return log(sigma / self.sigma_data) * 0.25

    def preconditioned_network_forward(
        self,
        noised_atom_coords,
        sigma,
        network_condition_kwargs: dict,
        training: bool = True,
    ):
        """Outer layer of per-step diffusion noise prediction. 

        Gets called at every diffusion step by AtomDiffusion.sample() 
        as well as for training. Does some input prep - prepares the 
        'dimensionless' version of the noised coordinates and the 
        scaled t-hat/sigma used to create the Fourier time embedding. 

        Parameters
        ----------
        noised_atom_coords : torch.tensor of shape (chunk_size, 
        n_padded_atoms, 3)
            A 'chunk' of noised atom coordinates (several samples taken)
            from the 0th (multiplicity) dimension.

        sigma : float
            t_hat (sigma_tm * (gamma + 1)) during inference, 
            padded_sigmas during training.
            Noise schedules for training and inference are very
            different. See page 24 of supps.

        network_condition_args : dict
            multiplicity : int
                The number of diffusion samples.
            
            model_cache : dict

            Additionally contains s_trunk, z_trunk, s_inputs, feats, and  
            relative_position_encoding (see AtomDiffusion.sample() 
            docstring).

        Returns
        -------
        denoised_coords : torch.tensor of shape (chunk_size, 
        n_padded_atoms, 3)
            Model prediction of a chunk of fully denoised coordinates at
            'timestep' sigma, conditioned on network_condition_args.
        """
        batch, device = noised_atom_coords.shape[0], noised_atom_coords.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        padded_sigma = rearrange(sigma, "b -> b 1 1")

        net_out = self.score_model(
            r_noisy=self.c_in(padded_sigma) * noised_atom_coords, # 'Dimensionless' scaling, algorithm 20 step 2.
            times=self.c_noise(sigma), # Algorithm 21 step 8.
            **network_condition_kwargs,
        )

        denoised_coords = (
            self.c_skip(padded_sigma) * noised_atom_coords
            + self.c_out(padded_sigma) * net_out["r_update"]
        ) # Algorithm 20 step 8.
        return denoised_coords, net_out["token_a"]

    def sample_schedule(self, num_sampling_steps=None):
        """AF3 diffusion noise scheduler. See pg. 24 of AF3 supps. 

        Returns
        -------
        sigmas : torch.tensor of shape (num_sampling_steps, )
        """
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        inv_rho = 1 / self.rho # 1/7 by default.

        steps = torch.arange(
            num_sampling_steps, device=self.device, dtype=torch.float32
        )
        sigmas = (
            self.sigma_max**inv_rho
            + steps
            / (num_sampling_steps - 1)
            * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        sigmas = sigmas * self.sigma_data # self.sigma_data is 16 by default.

        sigmas = F.pad(sigmas, (0, 1), value=0.0)  # last step is sigma value of 0.
        return sigmas
    
    def set_force_parameters(self, batch_size, atom_mask, t_hat=1.0, temperature=300.0, bias_potential_path=None,
                             length_units="angstroms", energy_units="kilocalorie_per_mole",
                             temperature_units="kelvin", **network_condition_kwargs):
        """Sets the temperature to use for the model force.

        Parameters
        ----------
        temperature : float
            The temperature in Kelvin.
        """
        self.batch_size_force = batch_size
        self.atom_mask = atom_mask
        self.t_hat_force = t_hat
        self.model_force_temperature = temperature
        if bias_potential_path is not None:
            bias_potential = torch.jit.load(bias_potential_path).to(self.device)
            self.bias_potential = bias_potential
        else:
            self.bias_potential = None
        self.network_condition_kwargs_force = network_condition_kwargs

        self.temperature_units = getattr(unit, temperature_units)
        self.length_units = getattr(unit, length_units)
        self.energy_units = getattr(unit, energy_units)

        # boltz force is always in kT / angstroms, so convert to desired units

        # only need to multiply by avogadro if energy is in per mole units
        if self.energy_units.is_compatible(unit.kilojoule_per_mole):
            self.force_unit_conversion = (unit.BOLTZMANN_CONSTANT_kB * self.model_force_temperature * self.temperature_units * unit.AVOGADRO_CONSTANT_NA / unit.angstrom
                                      ).value_in_unit(self.energy_units / self.length_units)
        else:
            self.force_unit_conversion = (unit.BOLTZMANN_CONSTANT_kB * self.model_force_temperature * self.temperature_units / unit.angstrom
                                      ).value_in_unit(self.energy_units / self.length_units)
            
    # @torch.compile() #TODO: make sure this works with torch.compile
    def get_force(self, positions, batched_input_output=False):
        """Computes the forces acting on the given positions.

        Parameters
        ----------
        positions : torch.Tensor
            The atomic positions to compute forces for, of shape (batch * n_atoms, 3) if batched_input_output is False, else (batch, n_atoms, 3). Units of angstroms

        Returns
        -------
        torch.Tensor
            The computed forces, of shape (batch * n_atoms, 3) if batched_input_output is False, else (batch, n_atoms, 3). Units of kcal/(mol*angstrom)
        """
        assert self.model_force_temperature is not None, \
            "Model force temperature not set. Please call set_force_parameters()."
        assert self.network_condition_kwargs_force is not None, \
            "Model force network condition kwargs not set. Please call set_force_parameters()."
        assert self.t_hat_force is not None, \
            "Model force t_hat not set. Please call set_force_parameters()."

        # print(f"Getting forces for position shape {positions.shape}", flush=True)

        # Reshape input from (batch * n_atoms, 3) to (batch_size, n_atoms, 3)
        if not batched_input_output:
            positions_reshaped = rearrange(positions, '(batch atoms) dim -> batch atoms dim', batch=self.batch_size_force)
        else:
            positions_reshaped = positions
        # center the coordinates before padding
        positions_reshaped = positions_reshaped - reduce(positions_reshaped, 'batch atoms dim -> batch 1 dim', 'mean')
        n_atoms = positions_reshaped.shape[1]
        
        # Pad coordinates to match the padded tensor shape expected by the model
        # atom_mask's last dimension defines the model's padded atom count.
        padding_dim = self.atom_mask.shape[-1]
        rows_to_pad = padding_dim - n_atoms
        if rows_to_pad < 0:
            raise ValueError(
                f"positions contain more atoms ({n_atoms}) than the model's atom_mask padding ({padding_dim})."
            )
        positions_padded = F.pad(positions_reshaped, pad=(0, 0, 0, rows_to_pad))

        # Sanity check: ensure the padded tensor matches the atom mask width
        if positions_padded.shape[1] != padding_dim:
            raise RuntimeError(
                f"Padding failed: expected padded atom dimension {padding_dim}, got {positions_padded.shape[1]}"
            )

        # Random augmentation of current coordinates for score calcs.
        R, t = compute_random_augmentation(positions_padded.shape[0], device=positions_padded.device)
        positions_padded = (
            torch.einsum("bmd,bds->bms", positions_padded, R) + t
        )

        atom_coords_denoised, _ = self.preconditioned_network_forward(
                                    positions_padded,
                                    self.t_hat_force,
                                    training=False,
                                    network_condition_kwargs=self.network_condition_kwargs_force
                                )
        
        # Kabsch aligning the denoised coordinates to the augmented positions.
        # weighted_rigid_align aligns the first argument to the second.
        with torch.autocast("cuda", enabled=False):
            atom_coords_denoised = weighted_rigid_align(
                atom_coords_denoised.float(),
                positions_padded.float(),
                self.atom_mask.float(),
                self.atom_mask.float(),
            )

            positions_padded = positions_padded.to(atom_coords_denoised)
        
        # Compute (rotated) score from denoised coordinates
        score_rotated_padded = (atom_coords_denoised - positions_padded) / (self.t_hat_force ** 2)

        # Undo rotation from augmentation
        score_unrotated_padded = torch.einsum("bmd,bsd->bms", score_rotated_padded, R)

        # Unpad the score to get back to original n_atoms shape
        score = score_unrotated_padded[:, :n_atoms, :].contiguous()
        force = score * self.force_unit_conversion
        
        if self.bias_potential is not None:
            with torch.set_grad_enabled(True):
                bias_force = self.bias_potential(positions_reshaped)

            # print(f"Bias force avg magnitude: {torch.mean(torch.norm(bias_force, dim=-1))}", flush=True)
            nonzero_rows = torch.any(bias_force != 0.0, dim=-1)
            nonzero_bias = bias_force[nonzero_rows]
            tqdm.write(
                f"Bias force avg magnitude (nonzero): "
                f"{torch.mean(torch.norm(nonzero_bias, dim=-1))}"
            )
            tqdm.write(
                f"{torch.sum(nonzero_rows)/self.batch_size_force} atoms have nonzero bias force"
            )
            tqdm.write(
                f"Model force avg magnitude: "
                f"{torch.mean(torch.norm(force, dim=-1))}"
            )
            force += bias_force        


        # Reshape back to (batch * n_atoms, 3)
        if not batched_input_output:    
            force = rearrange(force, 'batch atoms dim -> (batch atoms) dim')
        return force
    
    def get_score_and_energy_kt(self, positions, beta):
        # returns the score (gradient of log probability) and energy in units of kT. Note that the energy is only the part of the energy from the model force, not including any bias potential.
        # beta should be in inverse units of the model energy
        return beta * self.get_force(positions, batched_input_output=True), None

    def sample(
        self,
        atom_mask,
        num_sampling_steps=None,
        multiplicity=1,
        max_parallel_samples=None,
        train_accumulate_token_repr=False,
        steering_args=None,
        ode_args=None,
        mode='predict_diff',
        **network_condition_kwargs,
    ):
        """Conditioned diffusion rollout for Boltz structure prediction.

        Gets called within Boltz1's forward pass following the 
        Pairformer stack. Offers two prediction modes - standard
        diffusion sampling or deterministic PFODE sampling.

        Parameters
        ----------
        atom_mask : torch.tensor of shape (1, padded_n_atoms)
            padded_n_atoms = ceil(n_atoms / atoms_per_window_queries)).

        \\*\\*network_condition_kwargs : dict
            s_trunk : torch.tensor of shape (1, n_tokens, c_s)
                Post-trunk token-level sequence representation. 
                c_s = 384

            z_trunk : torch.tensor of shape (1, n_tokens, n_tokens, c_z)
                Post-trunk token-level pairwise representation. 
                c_z = 128

            s_inputs : torch.tensor of shape (1, n_tokens, c_s) ?
                Pre-trunk token-level sequence representation?

            relative_position_encoding : torch.tensor of shape (1, 
            n_tokens, n_tokens, c_z) ?
                Pairwise relative token encodings.

            feats : dict

        ode_args : dict
            atol : float
                Absolute tolerance of ODE solver.

            rtol : float
                Relative tolerance of ODE solver. 
        
        mode : str
            Sets which sampling mode to use. Options: 'predict_diff', 
            'predict_pfode'.
        """
        if steering_args is not None and (steering_args["fk_steering"] or steering_args["guidance_update"]):
            potentials = get_potentials()
        if steering_args is not None and steering_args["fk_steering"]:
            multiplicity = multiplicity * steering_args["num_particles"]
            energy_traj = torch.empty((multiplicity, 0), device=self.device)
            resample_weights = torch.ones(multiplicity, device=self.device).reshape(
                -1, steering_args["num_particles"]
            )
        if steering_args is not None and steering_args["guidance_update"]:
            scaled_guidance_update = torch.zeros(
                (multiplicity, *atom_mask.shape[1:], 3),
                dtype=torch.float32,
                device=self.device,
            )

        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps) # 200 by default.
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0) # (multiplicity, next multiple of 32)

        shape = (*atom_mask.shape, 3)
        token_repr_shape = (multiplicity, network_condition_kwargs['feats']['token_index'].shape[1], 2 * self.token_s) 
        # (multiplicity, n_tokens, 2 * 384)

        # get the schedule, which is returned as (sigma, num_sampling_stepsgamma) tuple, and pair up with the next sigma and gamma
        sigmas = self.sample_schedule(num_sampling_steps)
        gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0) # gamma_min=1, gamma_0=0.8.
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))

        # atom position is noise at the beginning
        init_sigma = sigmas[0] # Highest noise level.
        atom_coords = init_sigma * torch.randn(shape, device=self.device) # (multiplicity, n_padded_atoms, 3)
        atom_coords_denoised = None
        model_cache = {} if self.use_inference_model_cache else None

        sample_ids = torch.arange(multiplicity).to(atom_coords.device)
        sample_ids_chunks = sample_ids.split(max_parallel_samples)
                    
        token_repr = None
        token_a = None

        # PFODE sampling.
        # ------------------------------------------------------------------- #
        if mode == 'predict_pfode':
            atom_coords_denoised = torch.zeros_like(atom_coords)
            sigma_min, sigma_max = sigmas[-2], sigmas[0] # Last sigma is 0.
            t = atom_coords.new_tensor([sigma_max, sigma_min]) # Going backwards in time.

            for sample_ids_chunk in tqdm(
                sample_ids_chunks, 
                desc='Predicting batches of structures with PFODE sampling',
                mininterval=10
            ):
                fevals = 0
                def ode_fn(sigma, x):
                    # x: torch.tensor of shape (multiplicity, num_padded_atoms, 3)
                    nonlocal fevals
                    denoised, _ = self.preconditioned_network_forward(
                        x,
                        sigma.item(),
                        training=False,
                        network_condition_kwargs=dict(
                            multiplicity=len(x),
                            model_cache={},
                            **network_condition_kwargs,
                        ),
                    )
                    fevals += 1
                    score = (denoised - x) / (sigma ** 2)
                    update = (score * -sigma).detach()
                    update = update.squeeze(0)
                    return update
                    
                sol = odeint(
                    ode_fn, 
                    atom_coords[sample_ids_chunk], 
                    t, 
                    atol=ode_args['atol'], 
                    rtol=ode_args['rtol'], 
                    method='dopri5' # RK45.
                )
                print('EVALS: ', fevals)
                atom_coords[sample_ids_chunk] = sol[-1] # (multiplicity, n_padded_atoms, 3)

        # Standard diffusion.
        # ------------------------------------------------------------------- #
        # gradually denoise. sigma_tm 'lags behind' sigma_t by one.
        else:
            for step_idx, (sigma_tm, sigma_t, gamma) in enumerate(sigmas_and_gammas):
                random_R, random_tr = compute_random_augmentation(
                    multiplicity, device=atom_coords.device, dtype=atom_coords.dtype
                )
                atom_coords = atom_coords - atom_coords.mean(dim=-2, keepdims=True) # Centering to origin.
                atom_coords = (
                    torch.einsum("bmd,bds->bms", atom_coords, random_R) + random_tr
                )

                if steering_args is not None and steering_args["guidance_update"] and scaled_guidance_update is not None:
                    scaled_guidance_update = torch.einsum(
                        "bmd,bds->bms", scaled_guidance_update, random_R
                    )

                sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()

                # Gamma upscales t_hat in earlier reverse diffusion steps and 
                # downscales it deeper into the reverse diffusion.
                # Algorithm 18.
                t_hat = sigma_tm * (1 + gamma)
                steering_t = 1.0 - (step_idx / num_sampling_steps)
                noise_var = self.noise_scale**2 * (t_hat**2 - sigma_tm**2) # noise_scale=1.003
                eps = sqrt(noise_var) * torch.randn(shape, device=self.device)
                atom_coords_noisy = atom_coords + eps # eps is xi in AF3. 

                with torch.no_grad():
                    atom_coords_denoised = torch.zeros_like(atom_coords_noisy)
                    token_a = torch.zeros(token_repr_shape).to(atom_coords_noisy)

                    for sample_ids_chunk in tqdm(
                        sample_ids_chunks, 
                        desc='Predicting batches of structures with diffusion',
                        mininterval=10
                    ):
                        atom_coords_denoised_chunk, token_a_chunk = \
                            self.preconditioned_network_forward(
                                atom_coords_noisy[sample_ids_chunk],
                                t_hat,
                                training=False,
                                network_condition_kwargs=dict(
                                    multiplicity=sample_ids_chunk.numel(),
                                    model_cache=model_cache,
                                    **network_condition_kwargs,
                                ),
                            )
                        atom_coords_denoised[sample_ids_chunk] = atom_coords_denoised_chunk
                        token_a[sample_ids_chunk] = token_a_chunk
                    
                    # After the for loop, we have:
                    # atom_coords_denoised : torch.tensor of shape (multiplicity, n_padded_atoms, 3)
                    # token_a : torch.tensor of shape (multiplicity, n_tokens, 2 * 384)

                    if steering_args is not None and steering_args["fk_steering"] and (
                        (
                            step_idx % steering_args["fk_resampling_interval"] == 0
                            and noise_var > 0
                        )
                        or step_idx == num_sampling_steps - 1
                    ):
                        # Compute energy of x_0 prediction
                        energy = torch.zeros(multiplicity, device=self.device)
                        for potential in potentials:
                            parameters = potential.compute_parameters(steering_t)
                            if parameters["resampling_weight"] > 0:
                                component_energy = potential.compute(
                                    atom_coords_denoised,
                                    network_condition_kwargs["feats"],
                                    parameters,
                                )
                                energy += parameters["resampling_weight"] * component_energy
                        energy_traj = torch.cat((energy_traj, energy.unsqueeze(1)), dim=1)

                        # Compute log G values
                        if step_idx == 0:
                            log_G = -1 * energy
                        else:
                            log_G = energy_traj[:, -2] - energy_traj[:, -1]

                        # Compute ll difference between guided and unguided transition distribution
                        if steering_args["guidance_update"] and noise_var > 0:
                            ll_difference = (
                                eps**2 - (eps + scaled_guidance_update) ** 2
                            ).sum(dim=(-1, -2)) / (2 * noise_var)
                        else:
                            ll_difference = torch.zeros_like(energy)

                        # Compute resampling weights
                        resample_weights = F.softmax(
                            (ll_difference + steering_args["fk_lambda"] * log_G).reshape(
                                -1, steering_args["num_particles"]
                            ),
                            dim=1,
                        )

                    # Compute guidance update to x_0 prediction
                    if (
                        steering_args is not None and 
                        steering_args["guidance_update"]
                        and step_idx < num_sampling_steps - 1
                    ):
                        guidance_update = torch.zeros_like(atom_coords_denoised)
                        for guidance_step in range(steering_args["num_gd_steps"]):
                            energy_gradient = torch.zeros_like(atom_coords_denoised)
                            for potential in potentials:
                                parameters = potential.compute_parameters(steering_t)
                                if (
                                    parameters["guidance_weight"] > 0
                                    and (guidance_step) % parameters["guidance_interval"]
                                    == 0
                                ):
                                    energy_gradient += parameters[
                                        "guidance_weight"
                                    ] * potential.compute_gradient(
                                        atom_coords_denoised + guidance_update,
                                        network_condition_kwargs["feats"],
                                        parameters,
                                    )
                            guidance_update -= energy_gradient
                        atom_coords_denoised += guidance_update
                        scaled_guidance_update = (
                            guidance_update
                            * -1
                            * self.step_scale
                            * (sigma_t - t_hat)
                            / t_hat
                        )

                    if steering_args is not None and steering_args["fk_steering"] and (
                        (
                            step_idx % steering_args["fk_resampling_interval"] == 0
                            and noise_var > 0
                        )
                        or step_idx == num_sampling_steps - 1
                    ):
                        resample_indices = (
                            torch.multinomial(
                                resample_weights,
                                resample_weights.shape[1]
                                if step_idx < num_sampling_steps - 1
                                else 1,
                                replacement=True,
                            )
                            + resample_weights.shape[1]
                            * torch.arange(
                                resample_weights.shape[0], device=resample_weights.device
                            ).unsqueeze(-1)
                        ).flatten()

                        atom_coords = atom_coords[resample_indices]
                        atom_coords_noisy = atom_coords_noisy[resample_indices]
                        atom_mask = atom_mask[resample_indices]
                        if atom_coords_denoised is not None:
                            atom_coords_denoised = atom_coords_denoised[resample_indices]
                        energy_traj = energy_traj[resample_indices]
                        if steering_args["guidance_update"]:
                            scaled_guidance_update = scaled_guidance_update[
                                resample_indices
                            ]
                        if token_repr is not None:
                            token_repr = token_repr[resample_indices]
                        if token_a is not None:
                            token_a = token_a[resample_indices]

                if self.accumulate_token_repr:
                    if token_repr is None:
                        token_repr = torch.zeros_like(token_a)

                    with torch.set_grad_enabled(train_accumulate_token_repr):
                        sigma = torch.full(
                            (atom_coords_denoised.shape[0],),
                            t_hat,
                            device=atom_coords_denoised.device,
                        )
                        token_repr = self.out_token_feat_update(
                            times=self.c_noise(sigma), acc_a=token_repr, next_a=token_a
                        )

                # Perform rigid Kabsch alignment between noisy and denoised 
                # coordinates prior to interpolation.
                if self.alignment_reverse_diff:
                    with torch.autocast("cuda", enabled=False):
                        atom_coords_noisy = weighted_rigid_align(
                            atom_coords_noisy.float(),
                            atom_coords_denoised.float(),
                            atom_mask.float(),
                            atom_mask.float(),
                        )

                    atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)

                denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat # delta in algorithm 18 step 9.
                atom_coords_next = (
                    atom_coords_noisy
                    + self.step_scale * (sigma_t - t_hat) * denoised_over_sigma
                )

                atom_coords = atom_coords_next

        return dict(sample_atom_coords=atom_coords, diff_token_repr=token_repr)

    def loss_weight(self, sigma):
        return (sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data) ** 2)

    def noise_distribution(self, batch_size):
        return (
            self.sigma_data
            * (
                self.P_mean
                + self.P_std * torch.randn((batch_size,), device=self.device)
            ).exp()
        )

    def forward(
        self,
        s_inputs,
        s_trunk,
        z_trunk,
        relative_position_encoding,
        feats,
        multiplicity=1,
    ):
        # training diffusion step
        batch_size = feats["coords"].shape[0]

        if self.synchronize_sigmas:
            sigmas = self.noise_distribution(batch_size).repeat_interleave(
                multiplicity, 0
            )
        else:
            sigmas = self.noise_distribution(batch_size * multiplicity)
        padded_sigmas = rearrange(sigmas, "b -> b 1 1")

        atom_coords = feats["coords"]
        B, N, L = atom_coords.shape[0:3]
        atom_coords = atom_coords.reshape(B * N, L, 3)
        atom_coords = atom_coords.repeat_interleave(multiplicity // N, 0)
        feats["coords"] = atom_coords

        atom_mask = feats["atom_pad_mask"]
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        atom_coords = center_random_augmentation(
            atom_coords, atom_mask, augmentation=self.coordinate_augmentation
        )

        noise = torch.randn_like(atom_coords)
        noised_atom_coords = atom_coords + padded_sigmas * noise # (padded_sigmas ** 2 * I)

        denoised_atom_coords, _ = self.preconditioned_network_forward(
            noised_atom_coords,
            sigmas,
            training=True,
            network_condition_kwargs=dict(
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                relative_position_encoding=relative_position_encoding,
                feats=feats,
                multiplicity=multiplicity,
            ),
        )

        return dict(
            noised_atom_coords=noised_atom_coords,
            denoised_atom_coords=denoised_atom_coords,
            sigmas=sigmas,
            aligned_true_atom_coords=atom_coords,
        )

    def compute_loss(
        self,
        feats,
        out_dict,
        add_smooth_lddt_loss=True,
        nucleotide_loss_weight=5.0,
        ligand_loss_weight=10.0,
        multiplicity=1,
    ):
        denoised_atom_coords = out_dict["denoised_atom_coords"]
        noised_atom_coords = out_dict["noised_atom_coords"]
        sigmas = out_dict["sigmas"]

        resolved_atom_mask = feats["atom_resolved_mask"]
        resolved_atom_mask = resolved_atom_mask.repeat_interleave(multiplicity, 0)

        align_weights = noised_atom_coords.new_ones(noised_atom_coords.shape[:2])
        atom_type = (
            torch.bmm(
                feats["atom_to_token"].float(), feats["mol_type"].unsqueeze(-1).float()
            )
            .squeeze(-1)
            .long()
        )
        atom_type_mult = atom_type.repeat_interleave(multiplicity, 0)

        align_weights = align_weights * (
            1
            + nucleotide_loss_weight
            * (
                torch.eq(atom_type_mult, const.chain_type_ids["DNA"]).float()
                + torch.eq(atom_type_mult, const.chain_type_ids["RNA"]).float()
            )
            + ligand_loss_weight
            * torch.eq(atom_type_mult, const.chain_type_ids["NONPOLYMER"]).float()
        )

        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            atom_coords = out_dict["aligned_true_atom_coords"]
            atom_coords_aligned_ground_truth = weighted_rigid_align(
                atom_coords.detach().float(),
                denoised_atom_coords.detach().float(),
                align_weights.detach().float(),
                mask=resolved_atom_mask.detach().float(),
            )

        # Cast back
        atom_coords_aligned_ground_truth = atom_coords_aligned_ground_truth.to(
            denoised_atom_coords
        )

        # weighted MSE loss of denoised atom positions
        mse_loss = ((denoised_atom_coords - atom_coords_aligned_ground_truth) ** 2).sum(
            dim=-1
        )
        mse_loss = torch.sum(
            mse_loss * align_weights * resolved_atom_mask, dim=-1
        ) / torch.sum(3 * align_weights * resolved_atom_mask, dim=-1)

        # weight by sigma factor
        loss_weights = self.loss_weight(sigmas)
        mse_loss = (mse_loss * loss_weights).mean()

        total_loss = mse_loss

        # proposed auxiliary smooth lddt loss
        lddt_loss = self.zero
        if add_smooth_lddt_loss:
            lddt_loss = smooth_lddt_loss(
                denoised_atom_coords,
                feats["coords"],
                torch.eq(atom_type, const.chain_type_ids["DNA"]).float()
                + torch.eq(atom_type, const.chain_type_ids["RNA"]).float(),
                coords_mask=feats["atom_resolved_mask"],
                multiplicity=multiplicity,
            )

            total_loss = total_loss + lddt_loss

        loss_breakdown = dict(
            mse_loss=mse_loss,
            smooth_lddt_loss=lddt_loss,
        )

        return dict(loss=total_loss, loss_breakdown=loss_breakdown)
