# Quantization-Aware Training (QAT) fine-tuning for the VPT foundation model.
#
# This is a QAT variant of `behavioural_cloning.py`. It is *not* PTQ:
# instead of quantizing an already-trained model after the fact, we insert
# torchao fake-quantization into the policy's nn.Linear layers BEFORE training,
# run the normal behavioural-cloning loop so the weights learn to tolerate INT8
# rounding error, then convert to a real quantized model and save it.
#
# WHAT CHANGED vs behavioural_cloning.py
#   * LEARNING_RATE lowered to ~1/10 (QAT is a light fine-tune of trained weights).
#   * EPOCHS exposed at the top (QAT usually needs only a fraction of an epoch).
#   * After loading weights, fake-quant is inserted into the policy via torchao
#     `quantize_` + IntXQuantizationAwareTrainingConfig (eager mode, nn.Linear
#     only -- NO FX graph-mode tracing, which the recurrent transformer cannot do).
#   * The optimizer is built AFTER fake-quant insertion so it sees the right params.
#   * After training: strip fake-quant, move to CPU, apply real INT8 quantization.
#   * The BC training loop itself (gradient accumulation one sample at a time,
#     per-episode hidden-state tracking) is copied verbatim.
#
# OUTPUT
#   The saved `--out-weights` is an INT8 dynamic-activation / int8-weight model
#   (Int8DynamicActivationInt8WeightConfig) intended for CPU inference, matching
#   the INT8 paths in quantization_experiment.py.
#
# torchao API USED
#   torchao==0.11.0 (last release that supports this env's Python 3.9; the newer
#   0.17.0 cannot be imported on Python 3.9 because torchao/optim uses PEP-604
#   `X | None` syntax). The QAT symbols below are eager-mode `quantize_` configs:
#       from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
#       from torchao.quantization.qat import (
#           FakeQuantizeConfig,
#           IntXQuantizationAwareTrainingConfig,
#           FromIntXQuantizationAwareTrainingConfig,
#       )
#
# HOW TO RUN
#   python behavioural_cloning_qat.py \
#       --data-dir   path/to/MineRLBasaltFindCave-recordings \
#       --in-model   foundation-model-1x.model \
#       --in-weights foundation-model-1x.weights \
#       --out-weights foundation-model-1x-qat-int8.weights
#
# NOTE: like behavioural_cloning.py, this is illustrative fine-tuning code, not
#       the original VPT training pipeline. It trains one step at a time to fit
#       in small VRAM (tested target: RTX 3050 laptop, 4 GB).

from argparse import ArgumentParser
import pickle
import time

import gym
import minerl
import torch
import torch as th
import numpy as np

from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
from torchao.quantization.qat import (
    FakeQuantizeConfig,
    IntXQuantizationAwareTrainingConfig,
    FromIntXQuantizationAwareTrainingConfig,
)

from agent import PI_HEAD_KWARGS, MineRLAgent
from data_loader import DataLoader
from lib.tree_util import tree_map

# QAT often converges in a fraction of an epoch; lower this freely.
EPOCHS = 2
# Needs to be <= number of videos
BATCH_SIZE = 8
# Ideally more than batch size to create
# variation in datasets (otherwise, you will
# get a bunch of consecutive samples)
# Decrease this (and batch_size) if you run out of memory
N_WORKERS = 12
DEVICE = "cuda"

LOSS_REPORT_RATE = 100

# QAT light-fine-tune learning rate: ~1/10 of the BC value (behavioural_cloning.py
# uses 0.000181). We are only nudging already-trained weights to tolerate INT8.
LEARNING_RATE = 1.81e-5
WEIGHT_DECAY = 0.039428
MAX_GRAD_NORM = 5.0

def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs

def behavioural_cloning_qat_train(data_dir, in_model, in_weights, out_weights):
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)

    # To create model with the right environment.
    # All basalt environments have the same settings, so any of them works here
    env = gym.make("MineRLBasaltFindCave-v0")
    agent = MineRLAgent(env, device=DEVICE, policy_kwargs=agent_policy_kwargs, pi_head_kwargs=agent_pi_head_kwargs)
    agent.load_weights(in_weights)
    env.close()

    policy = agent.policy

    # --- Insert fake-quantization (QAT) -------------------------------------
    # Eager-mode torchao: targets nn.Linear only, no graph tracing. Activations
    # are quantized per-token (asymmetric), weights per-channel (symmetric).
    # per_channel (not group_size=N) matches the per-row weight quant that the
    # final Int8DynamicActivationInt8WeightConfig applies, and avoids group-size
    # divisibility errors on Linear layers whose in_features isn't a multiple of N.
    print("Inserting fake-quant")
    activation_config = FakeQuantizeConfig(torch.int8, "per_token", is_symmetric=False)
    weight_config     = FakeQuantizeConfig(torch.int8, "per_channel")
    quantize_(policy, IntXQuantizationAwareTrainingConfig(activation_config, weight_config))

    # Build the optimizer AFTER fake-quant insertion so it captures the
    # post-transform parameters (the quantize_ call swaps Linear modules).
    trainable_parameters = list(policy.parameters())

    # Parameters taken from the OpenAI VPT paper
    optimizer = th.optim.Adam(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    data_loader = DataLoader(
        dataset_dir=data_dir,
        n_workers=N_WORKERS,
        batch_size=BATCH_SIZE,
        n_epochs=EPOCHS
    )

    start_time = time.time()

    # Keep track of the hidden state per episode/trajectory.
    # DataLoader provides unique id for each episode, which will
    # be different even for the same trajectory when it is loaded
    # up again
    episode_hidden_states = {}
    dummy_first = th.from_numpy(np.array((False,))).to(DEVICE)

    # NOTE: no torch.autocast(bf16) wrapper here. torchao fake-quantizers compute
    # scales / zero-points in fp32, and mixing autocast bf16 activations with the
    # fake_quantize math can cause dtype mismatches. The one-sample-at-a-time
    # gradient accumulation below already keeps VRAM within the 4 GB budget.
    print("Training (QAT)")
    loss_sum = 0
    for batch_i, (batch_images, batch_actions, batch_episode_id) in enumerate(data_loader):
        batch_loss = 0
        for image, action, episode_id in zip(batch_images, batch_actions, batch_episode_id):
            agent_action = agent._env_action_to_agent(action, to_torch=True, check_if_null=True)
            if agent_action is None:
                # Action was null
                continue

            agent_obs = agent._env_obs_to_agent({"pov": image})
            if episode_id not in episode_hidden_states:
                # TODO need to clean up this hidden state after worker is done with the work item.
                #      Leaks memory, but not tooooo much at these scales (will be a problem later).
                episode_hidden_states[episode_id] = policy.initial_state(1)
            agent_state = episode_hidden_states[episode_id]

            pi_distribution, v_prediction, new_agent_state = policy.get_output_for_observation(
                agent_obs,
                agent_state,
                dummy_first
            )

            log_prob  = policy.get_logprob_of_action(pi_distribution, agent_action)

            # Make sure we do not try to backprop through sequence
            # (fails with current accumulation)
            new_agent_state = tree_map(lambda x: x.detach(), new_agent_state)
            episode_hidden_states[episode_id] = new_agent_state

            # Finally, update the agent to increase the probability of the
            # taken action.
            # Remember to take mean over batch losses
            loss = -log_prob / BATCH_SIZE
            batch_loss += loss.item()
            loss.backward()

        th.nn.utils.clip_grad_norm_(trainable_parameters, MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad()

        loss_sum += batch_loss
        if batch_i % LOSS_REPORT_RATE == 0:
            time_since_start = time.time() - start_time
            print(f"Time: {time_since_start:.2f}, Batches: {batch_i}, Avrg loss: {loss_sum / LOSS_REPORT_RATE:.4f}")
            loss_sum = 0

    # --- Strip fake-quant and convert to a real INT8 model ------------------
    # Remove the fake-quant observers, restoring plain nn.Linear modules whose
    # weights have been adapted to INT8 during training.
    print("Stripping fake-quant")
    quantize_(policy, FromIntXQuantizationAwareTrainingConfig())

    # Apply the real quantization on CPU (the INT8 paths in
    # quantization_experiment.py all run on CPU).
    print("Converting to INT8 (CPU)")
    policy = policy.to("cpu")
    quantize_(policy, Int8DynamicActivationInt8WeightConfig())

    state_dict = policy.state_dict()
    th.save(state_dict, out_weights)
    print(f"Saved to {out_weights}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the directory containing recordings to be trained on")
    parser.add_argument("--in-model", required=True, type=str, help="Path to the .model file to be finetuned")
    parser.add_argument("--in-weights", required=True, type=str, help="Path to the .weights file to be finetuned")
    parser.add_argument("--out-weights", required=True, type=str, help="Path where finetuned weights will be saved")

    args = parser.parse_args()
    behavioural_cloning_qat_train(args.data_dir, args.in_model, args.in_weights, args.out_weights)
