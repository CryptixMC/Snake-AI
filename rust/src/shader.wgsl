// Snake AI — batch neural network inference
// Architecture: 20 → 16 → 4  (ReLU hidden, argmax output)
// Each workgroup thread handles one individual.

const IN_SIZE:     u32 = 20u;
const H_SIZE:      u32 = 64u;
const OUT_SIZE:    u32 = 4u;
const PARAM_COUNT: u32 = 1604u;
const W1_END:      u32 = 1280u;  // IN * H
const B1_END:      u32 = 1344u;  // W1_END + H
const W2_END:      u32 = 1600u;  // B1_END + H * OUT
// b2: W2_END .. W2_END+OUT

@group(0) @binding(0) var<storage, read>       all_params : array<f32>;  // N × PARAM_COUNT
@group(0) @binding(1) var<storage, read>       all_obs    : array<f32>;  // N × IN_SIZE
@group(0) @binding(2) var<storage, read_write> actions    : array<u32>;  // N

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&actions) { return; }

    let pb = i * PARAM_COUNT;   // param base for individual i
    let ob = i * IN_SIZE;       // obs base

    // Layer 1: relu(W1ᵀ x + b1)
    var h: array<f32, 64>;
    for (var j: u32 = 0u; j < H_SIZE; j++) {
        var s = all_params[pb + W1_END + j];           // b1[j]
        for (var k: u32 = 0u; k < IN_SIZE; k++) {
            s += all_obs[ob + k] * all_params[pb + k * H_SIZE + j];
        }
        h[j] = max(0.0, s);
    }

    // Layer 2: argmax(W2ᵀ h + b2)
    var best_action: u32 = 0u;
    var best_val:    f32 = -1.0e30;
    for (var k: u32 = 0u; k < OUT_SIZE; k++) {
        var s = all_params[pb + W2_END + k];           // b2[k]
        for (var j: u32 = 0u; j < H_SIZE; j++) {
            s += h[j] * all_params[pb + B1_END + j * OUT_SIZE + k];
        }
        if s > best_val {
            best_val    = s;
            best_action = k;
        }
    }

    actions[i] = best_action;
}
