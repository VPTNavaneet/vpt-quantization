import io
import os
import sys
import time
import pickle
import contextlib

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from gym import spaces

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

def _asset(name):
    return os.path.join(_HERE, name)

from agent import MineRLAgent, ENV_KWARGS, TARGET_ACTION_SPACE

class _MockTask:
    fov_range = [70, 70]
    gamma_range = [2, 2]
    guiscale_range = [1, 1]
    resolution = [640, 360]
    cursor_size_range = [16.0, 16.0]

class _MockEnv:
    task = _MockTask()
    action_space = spaces.Dict(TARGET_ACTION_SPACE)

print("Loading model config...")
with open(_asset("foundation-model-1x.model"), "rb") as f:
    model_config = pickle.load(f)

print("Loading weights (once, reused for all benchmarks)...")
_WEIGHTS = torch.load(_asset("foundation-model-1x.weights"), map_location="cpu")
_DUMMY_OBS = {"pov": torch.randint(0, 256, (128, 128, 3), dtype=torch.uint8).numpy()}

# ── helpers ──────────────────────────────────────────────────────────────────

def load_agent(device):
    policy_kwargs = model_config["model"]["args"]["net"]["args"]
    pi_head_kwargs = dict(model_config["model"]["args"]["pi_head_opts"])
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(_MockEnv(), device=device,
                        policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.policy.load_state_dict(_WEIGHTS, strict=False)
    agent.reset()
    return agent

def _cast_state(state, dtype):
    if isinstance(state, torch.Tensor):
        return state.to(dtype=dtype) if state.is_floating_point() else state
    elif isinstance(state, (list, tuple)):
        return type(state)(_cast_state(s, dtype) for s in state)
    return state

def get_model_size_mb(model):
    """Serialize state_dict to BytesIO to get accurate size (handles quantized models)."""
    try:
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        return buf.tell() / 1024 / 1024
    except Exception:
        return sum(p.nelement() * p.element_size() for p in model.parameters()) / 1024 / 1024

def benchmark_latency(agent, device, n_runs=20, amp_dtype=None):
    """Returns (avg_latency_ms, peak_vram_mb). peak_vram_mb is 0 for CPU."""
    is_cuda = device == "cuda"
    amp_ctx = (torch.autocast(device_type=device, dtype=amp_dtype)
               if amp_dtype is not None else contextlib.nullcontext())
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
    times = []
    with torch.no_grad(), amp_ctx:
        for i in range(1):
            note = " (compiling CUDA kernels, may take a few minutes...)" if is_cuda else ""
            print(f"  Warmup 1/1...{note}", flush=True)
            agent.get_action(_DUMMY_OBS)
            print("  Warmup 1/1 done", flush=True)
        if is_cuda:
            torch.cuda.synchronize()
        for _ in range(n_runs):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            agent.get_action(_DUMMY_OBS)
            if is_cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    avg_ms = sum(times) / len(times) * 1000
    vram_mb = torch.cuda.max_memory_allocated() / 1024 ** 2 if is_cuda else 0.0
    return avg_ms, vram_mb

# ── Results list ──────────────────────────────────────────────────────────────
# Each entry: (display_name, bits, device_label, size_mb, latency_ms, vram_mb)
results = []

# ── 1. FP32 (GPU) ─────────────────────────────────────────────────────────────
print("\n[1/6] Benchmarking FP32 (GPU)...")
agent = load_agent("cuda")
size = get_model_size_mb(agent.policy)
latency, vram = benchmark_latency(agent, "cuda")
print(f"  Size: {size:.1f} MB  Latency: {latency:.2f} ms  VRAM: {vram:.0f} MB")
results.append(("FP32", 32, "GPU", size, latency, vram))
del agent; torch.cuda.empty_cache()

# ── 2. BF16 (GPU) ─────────────────────────────────────────────────────────────
print("\n[2/6] Benchmarking BF16 (GPU)...")
agent = load_agent("cuda")
agent.policy = agent.policy.to(torch.bfloat16)
agent.hidden_state = _cast_state(agent.hidden_state, torch.bfloat16)
for m in agent.policy.modules():
    if hasattr(m, "dtype") and m.dtype == torch.float32:
        m.dtype = torch.bfloat16
size = get_model_size_mb(agent.policy)
latency, vram = benchmark_latency(agent, "cuda", amp_dtype=torch.bfloat16)
print(f"  Size: {size:.1f} MB  Latency: {latency:.2f} ms  VRAM: {vram:.0f} MB")
results.append(("BF16", 16, "GPU", size, latency, vram))
del agent; torch.cuda.empty_cache()

# ── 3. FP16 (GPU) ─────────────────────────────────────────────────────────────
print("\n[3/6] Benchmarking FP16 (GPU)...")
agent = load_agent("cuda")
agent.policy = agent.policy.half()
agent.hidden_state = _cast_state(agent.hidden_state, torch.float16)
for m in agent.policy.modules():
    if hasattr(m, "dtype") and m.dtype == torch.float32:
        m.dtype = torch.float16
size = get_model_size_mb(agent.policy)
latency, vram = benchmark_latency(agent, "cuda", amp_dtype=torch.float16)
print(f"  Size: {size:.1f} MB  Latency: {latency:.2f} ms  VRAM: {vram:.0f} MB")
results.append(("FP16", 16, "GPU", size, latency, vram))
del agent; torch.cuda.empty_cache()

# ── 4. INT8 Dynamic (CPU) ─────────────────────────────────────────────────────
print("\n[4/6] Benchmarking INT8 Dynamic (CPU)...")
agent = load_agent("cpu")
agent.policy = torch.quantization.quantize_dynamic(
    agent.policy, {torch.nn.Linear}, dtype=torch.qint8
)
size = get_model_size_mb(agent.policy)
latency, _ = benchmark_latency(agent, "cpu")
print(f"  Size: {size:.1f} MB  Latency: {latency:.2f} ms")
results.append(("INT8\nDynamic", 8, "CPU", size, latency, 0.0))
del agent

# ── 5. INT8 Static (CPU) ──────────────────────────────────────────────────────
print("\n[5/6] Benchmarking INT8 Static (CPU)...")
agent = load_agent("cpu")
_int8_static_ok = False
try:
    agent.policy.eval()
    agent.policy.qconfig = torch.ao.quantization.get_default_qconfig("x86")
    torch.ao.quantization.prepare(agent.policy, inplace=True)
    print("  Calibrating (10 passes)...", flush=True)
    with torch.no_grad():
        for _ in range(10):
            agent.get_action(_DUMMY_OBS)
    torch.ao.quantization.convert(agent.policy, inplace=True)
    size = get_model_size_mb(agent.policy)
    latency, _ = benchmark_latency(agent, "cpu")
    print(f"  Size: {size:.1f} MB  Latency: {latency:.2f} ms")
    results.append(("INT8\nStatic", 8, "CPU", size, latency, 0.0))
    _int8_static_ok = True
except Exception as e:
    print(f"  INT8 Static skipped ({type(e).__name__}: {e})")
del agent

# ── 6. INT4 Weight-Only (CPU) ─────────────────────────────────────────────────
# torch.quantization.quantize_dynamic only supports qint8/float16 for nn.Linear;
# quint4x2 is embedding-only in the legacy API. Use torchao when available.
print("\n[6/6] Benchmarking INT4 Weight-Only (CPU)...")
agent = load_agent("cpu")
try:
    from torchao.quantization import quantize_, int4_weight_only
    quantize_(agent.policy, int4_weight_only())
    size = get_model_size_mb(agent.policy)
    latency, _ = benchmark_latency(agent, "cpu")
    print(f"  Size: {size:.1f} MB  Latency: {latency:.2f} ms")
    results.append(("INT4\nW-Only", 4, "CPU", size, latency, 0.0))
except ImportError:
    print("  INT4 skipped (install torchao for INT4 weight-only quantization: pip install torchao)")
except Exception as e:
    print(f"  INT4 skipped ({type(e).__name__}: {e})")
del agent

# ── Results table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print(f"{'Format':<16} {'Bits':>5} {'Device':>6} {'Size(MB)':>10} {'Latency(ms)':>13} {'VRAM(MB)':>10}")
print("=" * 68)
for name, bits, dl, size_mb, lat_ms, vram_mb in results:
    n = name.replace("\n", " ")
    v = f"{vram_mb:.0f}" if vram_mb > 0 else "  CPU"
    print(f"{n:<16} {bits:>5} {dl:>6} {size_mb:>10.1f} {lat_ms:>13.2f} {v:>10}")
print("=" * 68)

# ── Plot (2×2) ────────────────────────────────────────────────────────────────
names     = [r[0] for r in results]
sizes     = [r[3] for r in results]
latencies = [r[4] for r in results]
vrams     = [r[5] for r in results]
devs      = [r[2] for r in results]

_GPU_PALETTE = ["#c0392b", "#e67e22", "#f39c12"]
_CPU_PALETTE = ["#2980b9", "#27ae60", "#8e44ad", "#16a085"]
gi = ci = 0
colors = []
for d in devs:
    if d == "GPU":
        colors.append(_GPU_PALETTE[gi % len(_GPU_PALETTE)]); gi += 1
    else:
        colors.append(_CPU_PALETTE[ci % len(_CPU_PALETTE)]); ci += 1

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
(ax1, ax2), (ax3, ax4) = axes

def _annotate_bars(ax, vals, fmt="{:.1f}"):
    for i, v in enumerate(vals):
        ax.text(i, v * 1.015 + 0.3, fmt.format(v), ha="center", fontsize=9, fontweight="bold")

# Panel 1 — Model size
ax1.bar(names, sizes, color=colors, edgecolor="white", linewidth=0.5)
_annotate_bars(ax1, sizes)
ax1.set_title("Model Size", fontsize=13, fontweight="bold")
ax1.set_ylabel("Size (MB)")
ax1.axhline(sizes[0], color="gray", linestyle="--", alpha=0.35, linewidth=1)

# Panel 2 — Inference latency
ax2.bar(names, latencies, color=colors, edgecolor="white", linewidth=0.5)
_annotate_bars(ax2, latencies)
ax2.set_title("Inference Latency", fontsize=13, fontweight="bold")
ax2.set_ylabel("Latency (ms)")

# Panel 3 — Peak GPU VRAM (0 for CPU variants)
ax3.bar(names, vrams, color=colors, edgecolor="white", linewidth=0.5)
for i, (v, d) in enumerate(zip(vrams, devs)):
    label = f"{v:.0f}" if d == "GPU" else "CPU"
    ax3.text(i, v + 20, label, ha="center", fontsize=9, fontweight="bold")
ax3.axhline(4096, color="red", linestyle="--", alpha=0.5, linewidth=1.2, label="4 GB VRAM limit")
ax3.set_title("Peak GPU VRAM During Inference", fontsize=13, fontweight="bold")
ax3.set_ylabel("VRAM (MB)")
ax3.legend(fontsize=9)

# Panel 4 — Size vs latency scatter
for r, c in zip(results, colors):
    name, bits, dl, size_mb, lat_ms, _ = r
    marker = "o" if dl == "GPU" else "s"
    ax4.scatter(size_mb, lat_ms, color=c, s=150, zorder=5, marker=marker)
    ax4.annotate(name.replace("\n", " "), (size_mb, lat_ms),
                 textcoords="offset points", xytext=(7, 4), fontsize=8)
ax4.set_xlabel("Model Size (MB)")
ax4.set_ylabel("Latency (ms)")
ax4.set_title("Size vs Speed Tradeoff", fontsize=13, fontweight="bold")
gpu_patch = mpatches.Patch(color="#c0392b", label="GPU (CUDA)  ●")
cpu_patch = mpatches.Patch(color="#2980b9", label="CPU (quantized)  ■")
ax4.legend(handles=[gpu_patch, cpu_patch], fontsize=9)

plt.suptitle(
    "OpenAI VPT Foundation-Model-1x — Quantization Benchmark\n"
    "RTX 3050 Laptop GPU (4 GB VRAM, BF16 supported)  ·  PyTorch 2.7.1+cu118",
    fontsize=12, fontweight="bold",
)
plt.tight_layout()
out_path = os.path.join(_HERE, "quantization_results.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved → {out_path}")
plt.show()
