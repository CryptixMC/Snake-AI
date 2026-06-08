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
compute shader → copy to staging buffer → `map_async` → poll → read → unmap:

```rust
// gpu.rs — one step of batch inference
pub fn infer(&self, obs_flat: &[f32]) -> Vec<usize> {
    self.queue.write_buffer(&self.obs_buf, 0, bytemuck::cast_slice(obs_flat));

    let mut enc = self.device.create_command_encoder(&CommandEncoderDescriptor::default());
    {
        let mut pass = enc.begin_compute_pass(&ComputePassDescriptor { .. });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups((self.n as u32 + 63) / 64, 1, 1);
    }
    enc.copy_buffer_to_buffer(&self.actions_buf, 0, &self.readback_buf, 0, ..);
    self.queue.submit(std::iter::once(enc.finish()));

    // Synchronous readback — blocks until GPU is done
    let slice = self.readback_buf.slice(..);
    slice.map_async(MapMode::Read, |_| {});
    self.device.poll(Maintain::Wait);
    // ... extract and return actions
}
```

The synchronous readback (`poll(Maintain::Wait)`) is a meaningful difference: the Rust version
stalls the CPU waiting for GPU output every game step. PyTorch's scheduler overlaps these
more efficiently.

**Portability trade-off:** `wgpu` targets Vulkan, Metal, DX12, and WebGPU — the same code
runs on Apple Silicon, Windows, and the browser. PyTorch ROCm is Linux/AMD-only.

### CPU Fallback

| | Python | Rust |
|-|--------|------|
| Mechanism | `device = "cpu"` | `rayon::par_iter()` |
| Explicit code | No — PyTorch handles it | Yes — separate `evaluate_cpu()` |
| Parallelism | PyTorch's CPU thread pool | One OS thread per individual |

```rust
// ga.rs — explicit CPU fallback path
pub fn step(&mut self) -> Stats {
    self.fitness = match &self.gpu {
        Some(g) => Self::evaluate_gpu(g, &self.params, self.grid_size),
        None    => Self::evaluate_cpu(&self.params, self.grid_size),
    };
```

---

## 4. Neural Network Definition

The architecture is identical (20→64→4, ReLU hidden, flat weight vector of 1604 f32s).
The implementations diverge on *how* that math is expressed.

### Python — PyTorch tensor ops

```python
# network.py
def forward_batch(params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    N = params.shape[0]
    w1 = params[:, 0:1280].view(N, 20, 64)
    b1 = params[:, 1280:1344]
    w2 = params[:, 1344:1600].view(N, 64, 4)
    b2 = params[:, 1600:]

    h = torch.relu(torch.bmm(x.unsqueeze(1), w1).squeeze(1) + b1)
    return torch.bmm(h.unsqueeze(1), w2).squeeze(1) + b2
```

All 500 snakes evaluated simultaneously. `torch.bmm` is a batched matrix multiply dispatched
to the GPU as a single operation.

### Rust — manual loops

```rust
// network.rs
pub fn forward(params: &[f32], x: &[f32]) -> [f32; OUT_SIZE] {
    let mut h = [0f32; H_SIZE];
    for j in 0..H_SIZE {
        let mut s = b1[j];
        for i in 0..IN_SIZE { s += x[i] * w1[i * H_SIZE + j]; }
        h[j] = s.max(0.0);
    }
    // ... output layer
}
```

One individual at a time on the CPU. Parallelism is via `rayon` (one thread per snake) or
offloaded entirely to the WGSL shader. **There is no autograd** — this implementation only
does inference, not gradient-based training.

---

## 5. Difficulty Curve

### Python

- **Getting started:** `pip install torch numpy pygame` — minutes
- **GPU access:** `device = torch.device("cuda")` — one line; ROCm needs an env var
- **Changing architecture:** Adjust slice indices in `network.py` — 5 lines
- **Debugging:** `print()` tensors, `torch.profiler`, Jupyter cells
- **Hidden risks:** Python loops over tensors are slow (the game step loop in `ga.py` is the
  bottleneck, not the GPU inference); the GIL prevents true thread-level parallelism

### Rust

- **Getting started:** Cargo handles dependencies, but `wgpu` pipeline setup is ~120 lines
  before you dispatch a single kernel
- **GPU access:** Write WGSL, manage bind groups, buffer staging, `map_async`, `poll` — all manual
- **Changing architecture:** Update `PARAM_COUNT`, index constants, and the WGSL shader — 3 files
- **Debugging:** `dbg!()`, `println!()`, but no equivalent to PyTorch's tensor inspection tools;
  GPU shader debugging requires external tools (RenderDoc, etc.)
- **Compile-time safety:** The borrow checker catches data races and buffer overruns before the
  program runs — this is real value, especially for the GPU buffer casting via `bytemuck`
- **`async` overhead:** `wgpu` is async-native; `pollster::block_on` is used to drive it
  synchronously, which adds a layer to understand

### Learning curve summary

```
Easy ←————————————————————————————————→ Hard

Python research prototype:  ████░░░░░░  (familiar ML API, GPU in 1 line)
Python production tuning:   ██████░░░░  (profiling, avoiding Python overhead)
Rust CPU parallel:          ██████░░░░  (ownership + rayon, no GPU complexity)
Rust + wgpu GPU:            █████████░  (shaders, buffer management, async)
```

---

## 6. Use Cases

### Reach for Python when…

- You're **prototyping or doing research** — change topology, loss function, or GA parameters in
  minutes without touching multiple files
- You need **autograd / gradient-based training** — PyTorch's backward pass is essential for
  anything beyond neuroevolution (RL policy gradients, supervised learning, fine-tuning)
- You want **ecosystem breadth** — Hugging Face, Gymnasium, Stable-Baselines3, JAX, and thousands
  of pretrained models are Python-first
- Your GPU is AMD — ROCm support in PyTorch is far more mature than in Rust ML crates
- You're evolving **large populations** — `torch.bmm` scales to tens of thousands of individuals
  without writing a line of GPU code

### Reach for Rust when…

- You need a **production inference engine** — no Python runtime, predictable latency, small binary
- You're **embedding ML in a native application** — game engine, CLI tool, embedded system
- You need **cross-platform GPU** — `wgpu` runs on Vulkan, Metal, DX12, and WebGPU; PyTorch ROCm
  is Linux-only
- The network is **fixed and small** — manual matmul is fine at this scale; `candle` or `burn`
  are worth adding for larger models
- **Memory and latency are critical** — Rust gives you direct control over allocation and avoids
  GC pauses or Python interpreter overhead
- You want **compile-time correctness guarantees** — the borrow checker and type system catch
  entire classes of bug (buffer overruns, data races) that Python only surfaces at runtime

### The hybrid pattern

A common production workflow: **prototype and train in Python**, then **export weights and run
inference in Rust**. Both projects save weights in a flat f32 binary format —
`best_snake.pt` (Python) and `best_snake.bin` (Rust) — so this bridge is already half-built.

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

The honest summary: **Python is where ML research happens; Rust is where ML gets productionized.**
These two Snake AI projects demonstrate that you can build the same thing in both — but the Rust
version requires 60% more code, manual GPU programming, and explicit CPU/GPU path selection in
exchange for portability, predictable performance, and memory safety.
