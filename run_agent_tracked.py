from argparse import ArgumentParser
import pickle

import torch
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival

from agent import MineRLAgent, ENV_KWARGS


def main(model, weights):
    env = HumanSurvival(**ENV_KWARGS).make()
    print("---Loading model---")
    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    print("---Launching MineRL environment (be patient)---")
    obs = env.reset()

    steps = 0
    while True:
        minerl_action = agent.get_action(obs)
        obs, reward, done, info = env.step(minerl_action)
        env.render()
        steps += 1
        if steps % 100 == 0:
            vram = torch.cuda.memory_allocated() / 1024**2
            print(f"  Step {steps} | reward: {reward:.3f} | VRAM: {vram:.0f} MB")


if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, required=True, help="Path to the '.model' file to be loaded.")

    args = parser.parse_args()

    main(args.model, args.weights)
