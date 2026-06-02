import pickle
import time
import torch
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from agent import MineRLAgent, ENV_KWARGS

MODEL  = "foundation-model-1x.model"
WEIGHTS = "foundation-model-1x.weights"

def update_module_dtypes(model, dtype=torch.bfloat16):
    for module in model.modules():
        if hasattr(module, "dtype") and isinstance(module.dtype, torch.dtype):
            module.dtype = dtype


def cast_state(state, dtype=torch.bfloat16):
    if isinstance(state, torch.Tensor):
        return state.to(dtype)
    elif isinstance(state, (list, tuple)):
        return type(state)(cast_state(s, dtype) for s in state)
    elif isinstance(state, dict):
        return {k: cast_state(v, dtype) for k, v in state.items()}
    return state


def main():
    print("---Loading model (BF16)---")
    agent_parameters = pickle.load(open(MODEL, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])

    print("---Launching MineRL environment (be patient)---")
    env = HumanSurvival(**ENV_KWARGS).make()
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(WEIGHTS)

    print("---Converting to BF16---")
    agent.policy = agent.policy.to(torch.bfloat16)
    update_module_dtypes(agent.policy)
    agent.hidden_state = cast_state(agent.hidden_state)
    size_mb = sum(p.nelement() * p.element_size()
                  for p in agent.policy.parameters()) / 1024 / 1024
    print(f"  Model size : {size_mb:.1f} MB")
    print(f"  VRAM used  : {torch.cuda.memory_allocated() / 1024**2:.0f} MB")

    episode = 0
    while True:
        episode += 1
        print(f"--- Episode {episode} (BF16) ---")
        try:
            obs = env.reset()
            done = False
            steps = 0
            while not done:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    minerl_action = agent.get_action(obs)
                obs, reward, done, info = env.step(minerl_action)
                env.render()
                steps += 1
                if steps % 100 == 0:
                    vram = torch.cuda.memory_allocated() / 1024**2
                    print(f"  Step {steps} | reward: {reward:.3f} | VRAM: {vram:.0f} MB")
                if "error" in info:
                    print("  Environment error, resetting...")
                    break
            print(f"  Episode ended after {steps} steps")
            agent.reset()
            agent.hidden_state = cast_state(agent.hidden_state)

        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(3)
            try:
                obs = env.reset()
            except Exception:
                env.close()
                time.sleep(5)
                env = HumanSurvival(**ENV_KWARGS).make()
                agent = MineRLAgent(env, policy_kwargs=policy_kwargs,
                                    pi_head_kwargs=pi_head_kwargs)
                agent.load_weights(WEIGHTS)

if __name__ == "__main__":
    main()