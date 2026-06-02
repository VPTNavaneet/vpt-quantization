import torch
import time
import matplotlib.pyplot as plt
import json
import os
import sys

# Add VPT repo to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import MineRLAgent, ENV_KWARGS

print("Loading model config...")
with open("foundation-model-1x.model") as f:
    model_config = json.load(f)

# ── helpers ──────────────────────────────────────────────────────────────────

def load_agent(device):
    agent_parameters = dict(
        policy_kwargs=model_config["model"]["args"]["net"]["args"],
        pi_head_kwargs=model_config["model"]["args"]["pi_head_kwargs"],
    )
    agent = MineRLAgent(device=device, **agent_parameters)
    agent.load_weights("foundation-model-1x.weights")
    return agent

def get_model_size_mb(model):
    total = sum(p.nelement() * p.element_size() for p in model.parameters())
    return total / 1024 / 1024

def benchmark_latency(agent, device, n_runs=30):
    # dummy frame: (1, 128, 128, 3) uint8  — matches MineRL obs format
    dummy_obs = {
        "pov": torch.randint(0, 255, (128, 128, 3), dtype=torch.uint8).numpy()
    }
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            agent.get_action(dummy_obs)
            times.append(time.perf_counter() - start)
    avg_ms = (sum(times) / len(times)) * 1000
    return avg_ms

# ── FP32 ─────────────────────────────────────────────────────────────────────
print("\n[1/3] Benchmarking FP32 (GPU)...")
agent_fp32 = load_agent(device="cuda")
size_fp32 = get_model_size_mb(agent_fp32.policy)
latency_fp32 = benchmark_latency(agent_fp32, "cuda")
print(f"  Size:    {size_fp32:.1f} MB")
print(f"  Latency: {latency_fp32:.2f} ms")
del agent_fp32
torch.cuda.empty_cache()

# ── FP16 ─────────────────────────────────────────────────────────────────────
print("\n[2/3] Benchmarking FP16 (GPU half precision)...")
agent_fp16 = load_agent(device="cuda")
agent_fp16.policy = agent_fp16.policy.half()
size_fp16 = get_model_size_mb(agent_fp16.policy)
latency_fp16 = benchmark_latency(agent_fp16, "cuda")
print(f"  Size:    {size_fp16:.1f} MB")
print(f"  Latency: {latency_fp16:.2f} ms")
del agent_fp16
torch.cuda.empty_cache()

# ── INT8 ─────────────────────────────────────────────────────────────────────
print("\n[3/3] Benchmarking INT8 (CPU dynamic quantization)...")
agent_int8 = load_agent(device="cpu")
agent_int8.policy = torch.quantization.quantize_dynamic(
    agent_int8.policy,
    {torch.nn.Linear},
    dtype=torch.qint8
)
size_int8 = get_model_size_mb(agent_int8.policy)
latency_int8 = benchmark_latency(agent_int8, "cpu")
print(f"  Size:    {size_int8:.1f} MB")
print(f"  Latency: {latency_int8:.2f} ms")

# ── Results table ─────────────────────────────────────────────────────────────
print("\n" + "="*50)
print(f"{'Format':<10} {'Size (MB)':>12} {'Latency (ms)':>14}")
print("="*50)
for name, size, lat in [
    ("FP32",  size_fp32,  latency_fp32),
    ("FP16",  size_fp16,  latency_fp16),
    ("INT8",  size_int8,  latency_int8),
]:
    print(f"{name:<10} {size:>12.1f} {lat:>14.2f}")
print("="*50)

# ── Plot ──────────────────────────────────────────────────────────────────────
labels   = ["FP32", "FP16", "INT8"]
sizes    = [size_fp32,    size_fp16,    size_int8]
latencies= [latency_fp32, latency_fp16, latency_int8]
colors   = ["#e74c3c", "#3498db", "#2ecc71"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.bar(labels, sizes, color=colors)
ax1.set_title("Model Size by Precision", fontsize=13)
ax1.set_ylabel("Size (MB)")
for i, v in enumerate(sizes):
    ax1.text(i, v + 1, f"{v:.1f}", ha="center", fontweight="bold")

ax2.bar(labels, latencies, color=colors)
ax2.set_title("Inference Latency by Precision", fontsize=13)
ax2.set_ylabel("Latency (ms)")
for i, v in enumerate(latencies):
    ax2.text(i, v + 0.5, f"{v:.2f}", ha="center", fontweight="bold")

plt.suptitle("VPT Model Quantization Comparison\nRTX 3050 Laptop GPU (4GB)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("quantization_results.png", dpi=150, bbox_inches="tight")
print("\nPlot saved as quantization_results.png")
plt.show()