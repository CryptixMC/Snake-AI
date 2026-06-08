// Architecture: 20 → 64 → 4
// Inputs: 8 wall + 8 body + 4 food directions (up/down/left/right, all >= 0)
// Flat weight order: W1 | b1 | W2 | b2

use rand_distr::{Distribution, Normal};

pub const IN_SIZE:  usize = 20;
pub const H_SIZE:   usize = 64;
pub const OUT_SIZE: usize = 4;

const W1_END: usize = IN_SIZE * H_SIZE;            // 1280
const B1_END: usize = W1_END + H_SIZE;             // 1344
const W2_END: usize = B1_END + H_SIZE * OUT_SIZE;  // 1600
pub const PARAM_COUNT: usize = W2_END + OUT_SIZE;  // 1604

pub fn forward(params: &[f32], x: &[f32]) -> [f32; OUT_SIZE] {
    let w1 = &params[..W1_END];
    let b1 = &params[W1_END..B1_END];
    let w2 = &params[B1_END..W2_END];
    let b2 = &params[W2_END..];

    let mut h = [0f32; H_SIZE];
    for j in 0..H_SIZE {
        let mut s = b1[j];
        for i in 0..IN_SIZE { s += x[i] * w1[i * H_SIZE + j]; }
        h[j] = s.max(0.0);
    }

    let mut out = [0f32; OUT_SIZE];
    for k in 0..OUT_SIZE {
        let mut s = b2[k];
        for j in 0..H_SIZE { s += h[j] * w2[j * OUT_SIZE + k]; }
        out[k] = s;
    }
    out
}

pub fn argmax(logits: &[f32; OUT_SIZE]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

pub struct ForwardResult {
    pub inputs: [f32; IN_SIZE],
    pub hidden: [f32; H_SIZE],
    pub output: [f32; OUT_SIZE],
    pub action: usize,
    pub w1: [[f32; H_SIZE]; IN_SIZE],
    pub w2: [[f32; OUT_SIZE]; H_SIZE],
}

pub fn forward_with_activations(params: &[f32], x: &[f32]) -> ForwardResult {
    let w1s = &params[..W1_END];
    let b1  = &params[W1_END..B1_END];
    let w2s = &params[B1_END..W2_END];
    let b2  = &params[W2_END..];

    let mut w1 = [[0f32; H_SIZE]; IN_SIZE];
    for i in 0..IN_SIZE {
        for j in 0..H_SIZE { w1[i][j] = w1s[i * H_SIZE + j]; }
    }

    let mut hidden = [0f32; H_SIZE];
    for j in 0..H_SIZE {
        let mut s = b1[j];
        for i in 0..IN_SIZE { s += x[i] * w1[i][j]; }
        hidden[j] = s.max(0.0);
    }

    let mut w2 = [[0f32; OUT_SIZE]; H_SIZE];
    for j in 0..H_SIZE {
        for k in 0..OUT_SIZE { w2[j][k] = w2s[j * OUT_SIZE + k]; }
    }

    let mut output = [0f32; OUT_SIZE];
    for k in 0..OUT_SIZE {
        let mut s = b2[k];
        for j in 0..H_SIZE { s += hidden[j] * w2[j][k]; }
        output[k] = s;
    }

    let mut inputs = [0f32; IN_SIZE];
    inputs.copy_from_slice(x);

    ForwardResult { inputs, hidden, output, action: argmax(&output), w1, w2 }
}

pub fn random_population(n: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    let w1_std = (2.0f32 / (IN_SIZE + H_SIZE) as f32).sqrt();
    let w2_std = (2.0f32 / (H_SIZE + OUT_SIZE) as f32).sqrt();
    let w1_dist = Normal::new(0.0f32, w1_std).unwrap();
    let w2_dist = Normal::new(0.0f32, w2_std).unwrap();

    (0..n)
        .map(|_| {
            let mut p = vec![0f32; PARAM_COUNT];
            for i in 0..W1_END        { p[i] = w1_dist.sample(&mut rng); }
            for i in B1_END..W2_END   { p[i] = w2_dist.sample(&mut rng); }
            p
        })
        .collect()
}
