# Snake AI

> **WARNING: This project was vibe coded.**
> Expect rough edges, inconsistent style, and decisions made on vibes rather than rigor.
> It exists purely as a testbed — not as a reference implementation.

---

A side-by-side neuroevolution experiment: the same Snake AI implemented in **Python** and **Rust**, evolving a neural network via a genetic algorithm. The goal is to compare how the two languages handle ML workloads — ergonomics, ecosystem, GPU access, and raw verbosity.

See [COMPARISON.md](COMPARISON.md) for a detailed breakdown of every meaningful difference.

---

## What it is

Both implementations:
- Evolve a fixed-topology neural network (20 → 64 → 4) using a genetic algorithm
- Use identical GA parameters and fitness function (`food_eaten² × steps_survived`)
- Support GPU-accelerated batch inference (PyTorch/ROCm in Python, wgpu/Vulkan in Rust)
- Can visualize the best agent playing in real time

The point is not to build the best Snake AI. The point is to see what it costs — in code, complexity, and effort — to do the same thing in each language.

---

## Python

**Stack:** PyTorch (ROCm), numpy, pygame

```sh
cd python
nix develop   # sets up HSA_OVERRIDE_GFX_VERSION and ROCm env
python train.py
python watch.py          # visualize best_snake.pt
```

The entire population is evaluated in a single batched GPU call per step via `torch.bmm`. No shader code, no buffer management — PyTorch handles it.

## Rust

**Stack:** wgpu (Vulkan), rayon, macroquad, clap

```sh
cd rust
nix develop   # sets VK_ICD_FILENAMES for RADV driver
cargo run --release -- train
cargo run --release -- watch    # visualize best_snake.bin
```

GPU path uses a hand-written WGSL compute shader. Falls back to `rayon` CPU parallelism if no Vulkan device is available.

---

## Hardware this was tested on

- AMD RX 6800 XT (RDNA2, gfx1030) via eGPU
- NixOS 26.05
- Python: ROCm via nixpkgs `rocmSupport = true`
- Rust: Vulkan via RADV (mesa drivers)

Both `flake.nix` files configure the necessary environment variables for this setup. Other hardware may need adjustments.

---

## TL;DR from the comparison

| | Python | Rust |
|---|---|---|
| Total LOC | ~775 | ~1,253 |
| GPU backend | PyTorch (`torch.bmm`) | WGSL shader + 241 LOC of buffer management |
| Time to prototype | Fast | Slow |
| Compile-time safety | No | Yes |
| Cross-platform GPU | No (ROCm = Linux/AMD only) | Yes (wgpu: Vulkan/Metal/DX12/WebGPU) |

**Python is where ML research happens. Rust is where ML gets productionized.**

Full analysis: [COMPARISON.md](COMPARISON.md)
