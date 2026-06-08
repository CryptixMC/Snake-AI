// Snake AI — batch neural network inference
// Architecture: 20 → 64 → 4  (ReLU hidden, argmax output)
// Each workgroup thread handles one individual.

const IN_SIZE:     u32 = 20u;
const H_SIZE:      u32 = 64u;
const OUT_SIZE:    u32 = 4u;
const PARAM_COUNT: u32 = 1604u;
const W1_END:      u32 = 1280u;
const B1_END:      u32 = 1344u;
const W2_END:      u32 = 1600u;

@group(0) @binding(0) var<storage, read>       all_params : array<f32>;
@group(0) @binding(1) var<storage, read>       all_obs    : array<f32>;
@group(0) @binding(2) var<storage, read_write> actions    : array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&actions) { return; }

    let pb = i * PARAM_COUNT;
    let ob = i * IN_SIZE;

    var h: array<f32, 64>;
    for (var j: u32 = 0u; j < H_SIZE; j++) {
        var s = all_params[pb + W1_END + j];
        for (var k: u32 = 0u; k < IN_SIZE; k++) {
            s += all_obs[ob + k] * all_params[pb + k * H_SIZE + j];
        }
        h[j] = max(0.0, s);
    }

    var best_action: u32 = 0u;
    var best_val:    f32 = -1.0e30;
    for (var k: u32 = 0u; k < OUT_SIZE; k++) {
        var s = all_params[pb + W2_END + k];
        for (var j: u32 = 0u; j < H_SIZE; j++) {
            s += h[j] * all_params[pb + B1_END + j * OUT_SIZE + k];
        }
        if s > best_val { best_val = s; best_action = k; }
    }

    actions[i] = best_action;
}
