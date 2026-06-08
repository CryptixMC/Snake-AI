# Python vs Rust for ML Projects
### A Case Study Using Snake AI Neuroevolution

Both implementations in this repo evolve a neural network to play Snake using a genetic algorithm.
They share identical architecture (20→64→4), identical GA parameters, and identical fitness functions —
making them a controlled comparison of *language and tooling*, not algorithm design.

---

## 1. Ecosystem & Available Tools

### Python

Python's ML ecosystem is the most mature in existence. A few key layers:

| Library | Role |
|---------|------|
| `torch` | Neural networks, GPU tensor ops, autograd, ROCm/CUDA backends |
| `numpy` | Array math, observation stacking |
| `pygame` | 2D visualization |
| `gymnasium` | Standard RL environment interface (not used here, but free to drop in) |
| `stable-baselines3` | Off-the-shelf RL algorithms (PPO, DQN, etc.) |
| `jax` | Alternative to PyTorch; functional, JIT-compiled, excellent for neuroevolution |
| `scikit-learn` | Classical ML |
| `jupyter` | Interactive notebooks for exploration |

The key insight is that **the ecosystem does the hard work for you**. A batched forward pass for
500 snakes simultaneously is two lines:

```python
# network.py — entire population in one GPU call
h = torch.relu(torch.bmm(x.unsqueeze(1), w1).squeeze(1) + b1)
return torch.bmm(h.unsqueeze(1), w2).squeeze(1) + b2
```

PyTorch handles memory layout, kernel dispatch, and GPU scheduling. You never touch a shader.

### Rust

Rust's ML ecosystem is nascent. The notable crates as of 2025:

| Crate | Role |
|-------|------|
| `wgpu` | Cross-platform GPU compute (Vulkan/Metal/DX12) via WGSL shaders |
| `rayon` | CPU data parallelism via `par_iter()` |
| `candle` (HuggingFace) | PyTorch-like tensor ops in pure Rust |
| `burn` | Modular ML framework with multiple backends |
| `tch-rs` | Rust bindings to libtorch (PyTorch's C++ core) |
| `macroquad` | Game loop + 2D rendering |
| `clap` | CLI argument parsing |

This project skips `candle`/`burn` entirely and does **manual matrix multiplication** because the
network is small and fixed-topology — the libraries add dependencies without adding value here.
For anything larger or trainable, you'd want `candle` or `tch-rs`.

The Rust GPU path requires writing compute shaders yourself in WGSL:

```wgsl
// shader.wgsl — hand-written, runs on the GPU
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    // Layer 1: relu(W1ᵀ x + b1)
    var h: array<f32, 64>;
    for (var j: u32 = 0u; j < H_SIZE; j++) {
        var s = all_params[pb + W1_END + j];
        for (var k: u32 = 0u; k < IN_SIZE; k++) {
            s += all_obs[ob + k] * all_params[pb + k * H_SIZE + j];
        }
        h[j] = max(0.0, s);
    }
    // ... Layer 2, argmax ...
}
```

That shader is 51 lines. The surrounding `gpu.rs` that manages buffers, pipelines, and readback is 190 lines.
Python replaces all 241 lines with `torch.bmm()`.

---

## 2. Lines of Code & Complexity

| Metric | Python | Rust |
|--------|--------|------|
| Total LOC | ~775 | ~1,253 |
| GPU backend code | 0 (PyTorch handles it) | 241 LOC (`gpu.rs` + `shader.wgsl`) |
| CPU fallback | `device = torch.device("cpu")` | Explicit `evaluate_cpu()` via `rayon` |
| NN definition | 18 lines (`network.py`) | 40 lines (`network.rs`) |
| Visualization | 307 lines (`visualize.py`) | 325 lines (`visualize.rs`) |

The 60% LOC gap is almost entirely GPU management. The game logic, GA, and NN
are comparable in size.

---

## 3. GPU Support

### Python — ROCm + PyTorch

AMD GPU access is configured in `flake.nix` with `rocmSupport = true` and a single
environment variable:

```
HSA_OVERRIDE_GFX_VERSION=10.3.0
```

Switching from GPU to CPU is one argument:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

The entire population is evaluated in a single batched GPU dispatch per game step. PyTorch
schedules the kernel, handles memory layout, and returns a tensor. No synchronization boilerplate.

### Rust — wgpu / Vulkan

The Rust path uses Vulkan via `wgpu`. Setup for an AMD RX 6800 XT on NixOS requires
the RADV ICD driver configured in `flake.nix`:

```nix
VK_ICD_FILENAMES = "${pkgs.mesa.drivers}/share/vulkan/icd.d/radeon_icd.x86_64.json";
```

Each inference step requires an explicit round-trip: upload observations → dispatch
compute shader → copy to staging buffer → `map_async` → poll → read → unmap.

The synchronous readback (`poll(Maintain::Wait)`) stalls the CPU waiting for GPU output every
game step. PyTorch's scheduler overlaps these more efficiently.

**Portability trade-off:** `wgpu` targets Vulkan, Metal, DX12, and WebGPU — the same code
runs on Apple Silicon, Windows, and the browser. PyTorch ROCm is Linux/AMD-only.

---

## 4. Neural Network Definition

The architecture is identical (20→64→4, ReLU hidden, flat weight vector of 1604 f32s).

### Python — PyTorch tensor ops

All 500 snakes evaluated simultaneously. `torch.bmm` is a batched matrix multiply dispatched
to the GPU as a single operation.

### Rust — manual loops

One individual at a time on the CPU. Parallelism is via `rayon` (one thread per snake) or
offloaded entirely to the WGSL shader. **There is no autograd** — this implementation only
does inference, not gradient-based training.

---

## 5. Difficulty Curve

```
Easy ←—————————————————————————→ Hard

Python research prototype:  ████░░░░░░  (familiar ML API, GPU in 1 line)
Python production tuning:   ██████░░░░  (profiling, avoiding Python overhead)
Rust CPU parallel:          ██████░░░░  (ownership + rayon, no GPU complexity)
Rust + wgpu GPU:            █████████░  (shaders, buffer management, async)
```

---

## 6. Use Cases

### Reach for Python when…

- You're **prototyping or doing research**
- You need **autograd / gradient-based training**
- You want **ecosystem breadth** — Hugging Face, Gymnasium, Stable-Baselines3, JAX
- Your GPU is AMD — ROCm support in PyTorch is far more mature than in Rust ML crates
- You're evolving **large populations** — `torch.bmm` scales without writing GPU code

### Reach for Rust when…

- You need a **production inference engine** — no Python runtime, predictable latency
- You're **embedding ML in a native application** — game engine, CLI tool, embedded system
- You need **cross-platform GPU** — `wgpu` runs on Vulkan, Metal, DX12, and WebGPU
- The network is **fixed and small** — manual matmul is fine at this scale
- **Memory and latency are critical**
- You want **compile-time correctness guarantees**

### The hybrid pattern

**Prototype and train in Python**, then **export weights and run inference in Rust**.
Both projects save weights in a flat f32 binary format so this bridge is already half-built.

---

## 7. Key Takeaways

| Question | Answer |
|----------|--------|
| Which is faster to write? | Python, by a wide margin |
| Which has more ML libraries? | Python — Rust's ecosystem is 5-10 years behind |
| Which runs faster (inference)? | Rust (no interpreter overhead), but Python+GPU often wins for large batches |
| Which is safer / more reliable? | Rust — compile-time guarantees for memory and concurrency |
| Which scales to training? | Python — Rust has no mature autograd |
| Which deploys more portably? | Rust (`wgpu` cross-platform) |
| Which should you learn first? | Python — the ecosystem will make you more productive immediately |

**Python is where ML research happens; Rust is where ML gets productionized.**
