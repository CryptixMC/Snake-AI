use rand::Rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

use crate::gpu::GpuInference;
use crate::network::{forward, argmax, random_population, IN_SIZE};
use crate::snake_env::SnakeEnv;

pub struct Stats {
    pub generation:   usize,
    pub best_fitness: f32,
    pub avg_fitness:  f32,
    pub best_score:   usize,
}

pub struct GeneticAlgorithm {
    pub params:          Vec<Vec<f32>>,
    pub fitness:         Vec<f32>,
    pub generation:      usize,
    pub ranked_params:   Vec<Vec<f32>>,

    pub population_size: usize,
    pub grid_size:       i32,
    elite_n:             usize,
    mutation_std:        f32,
    mutation_rate:       f32,
    crossover_rate:      f32,
    rank_probs:          Vec<f64>,
    gpu:                 Option<GpuInference>,
}

impl GeneticAlgorithm {
    pub fn new(
        population_size:    usize,
        grid_size:          i32,
        elite_frac:         f32,
        selection_pressure: f32,
        mutation_std:       f32,
        mutation_rate:      f32,
        crossover_rate:     f32,
        gpu:                Option<GpuInference>,
    ) -> Self {
        let elite_n    = ((population_size as f32 * elite_frac) as usize).max(1);
        let rank_probs = build_rank_probs(population_size, selection_pressure);

        Self {
            params:             random_population(population_size),
            fitness:            vec![0.0; population_size],
            generation:         0,
            ranked_params:      vec![],
            population_size,
            grid_size,
            elite_n,
            mutation_std,
            mutation_rate,
            crossover_rate,
            rank_probs,
            gpu,
        }
    }

    pub fn step(&mut self) -> Stats {
        self.fitness = match &self.gpu {
            Some(g) => Self::evaluate_gpu(g, &self.params, self.grid_size),
            None    => Self::evaluate_cpu(&self.params, self.grid_size),
        };
        self.generation += 1;

        // Sort indices best → worst
        let mut ranked: Vec<usize> = (0..self.population_size).collect();
        ranked.sort_unstable_by(|&a, &b| {
            self.fitness[b].partial_cmp(&self.fitness[a]).unwrap()
        });

        self.ranked_params = ranked.iter().map(|&i| self.params[i].clone()).collect();

        let mut new_params = self.params.clone();

        // Elites
        for (slot, &src) in ranked[..self.elite_n].iter().enumerate() {
            new_params[slot] = self.params[src].clone();
        }

        // Offspring
        let mut rng = rand::thread_rng();
        for i in self.elite_n..self.population_size {
            let child = if rng.gen::<f32>() < self.crossover_rate {
                let p1 = &self.params[self.rank_select(&ranked, &mut rng)];
                let p2 = &self.params[self.rank_select(&ranked, &mut rng)];
                crossover(p1, p2, &mut rng)
            } else {
                self.params[self.rank_select(&ranked, &mut rng)].clone()
            };
            new_params[i] = self.mutate(child, &mut rng);
        }

        self.params = new_params;

        let best_fitness = self.fitness[ranked[0]];
        Stats {
            generation:   self.generation,
            best_fitness,
            avg_fitness:  self.fitness.iter().sum::<f32>() / self.population_size as f32,
            best_score:   (best_fitness as usize) / 1000,
        }
    }

    pub fn top_n_params(&self, n: usize) -> Vec<Vec<f32>> {
        self.ranked_params[..n.min(self.ranked_params.len())].to_vec()
    }

    // ── Private ──────────────────────────────────────────────────────────────

    fn food_seq(grid_size: i32) -> Vec<(i32, i32)> {
        let mut rng = rand::thread_rng();
        (0..500).map(|_| (rng.gen_range(0..grid_size), rng.gen_range(0..grid_size))).collect()
    }

    fn fitness_of(env: &SnakeEnv, max_dist: f32) -> f32 {
        let (hr, hc) = env.snake[0];
        let (fr, fc) = env.food;
        let dist_bonus = max_dist - ((hr - fr).abs() + (hc - fc).abs()) as f32;
        if env.score > 0 {
            env.score as f32 * 1000.0 - env.steps as f32 * 0.1 + dist_bonus
        } else {
            dist_bonus
        }
    }

    /// GPU path: sequential game loop, GPU batch inference each step.
    fn evaluate_gpu(gpu: &GpuInference, params: &[Vec<f32>], grid_size: i32) -> Vec<f32> {
        let n        = params.len();
        let max_dist = ((grid_size - 1) * 2) as f32;
        let food_seq = Self::food_seq(grid_size);

        let mut envs: Vec<SnakeEnv> = (0..n)
            .map(|_| SnakeEnv::new(grid_size, food_seq.clone()))
            .collect();

        // Flat obs buffer: n × IN_SIZE, updated in-place each step.
        let mut obs_flat: Vec<f32> = envs.iter_mut()
            .flat_map(|e| e.reset())
            .collect();
        let mut alive = vec![true; n];

        gpu.upload_params(params);

        loop {
            if alive.iter().all(|&a| !a) { break; }

            let actions = gpu.infer(&obs_flat);

            for i in 0..n {
                if !alive[i] { continue; }
                let (new_obs, done) = envs[i].step(actions[i]);
                let base = i * IN_SIZE;
                obs_flat[base..base + IN_SIZE].copy_from_slice(&new_obs);
                if done { alive[i] = false; }
            }
        }

        envs.iter().map(|e| Self::fitness_of(e, max_dist)).collect()
    }

    /// CPU path: rayon parallel, one thread per individual.
    fn evaluate_cpu(params: &[Vec<f32>], grid_size: i32) -> Vec<f32> {
        let max_dist = ((grid_size - 1) * 2) as f32;
        let food_seq = Self::food_seq(grid_size);

        params.par_iter().map(|p| {
            let mut env = SnakeEnv::new(grid_size, food_seq.clone());
            let mut obs = env.reset();
            loop {
                let action = argmax(&forward(p, &obs));
                let (new_obs, done) = env.step(action);
                obs = new_obs;
                if done { break; }
            }
            Self::fitness_of(&env, max_dist)
        }).collect()
    }

    fn rank_select(&self, ranked: &[usize], rng: &mut impl Rng) -> usize {
        let r: f64 = rng.gen();
        let mut cum = 0.0f64;
        for (pos, p) in self.rank_probs.iter().enumerate() {
            cum += p;
            if r < cum {
                return ranked[pos];
            }
        }
        ranked[ranked.len() - 1]
    }

    fn mutate(&self, mut p: Vec<f32>, rng: &mut impl Rng) -> Vec<f32> {
        let fine_dist  = Normal::new(0.0f32, self.mutation_std).unwrap();
        let large_dist = Normal::new(0.0f32, self.mutation_std * 5.0).unwrap();
        for v in p.iter_mut() {
            if rng.gen::<f32>() < self.mutation_rate {
                *v += fine_dist.sample(rng);
            }
            // Rare large jumps (2%) for occasional exploration
            if rng.gen::<f32>() < 0.02 {
                *v += large_dist.sample(rng);
            }
        }
        p
    }
}

fn build_rank_probs(n: usize, selection_pressure: f32) -> Vec<f64> {
    let sp = selection_pressure as f64;
    let weights: Vec<f64> = (0..n)
        .map(|i| sp - (sp - 1.0) * i as f64 / (n - 1).max(1) as f64)
        .collect();
    let total: f64 = weights.iter().sum();
    weights.iter().map(|w| w / total).collect()
}

fn crossover(p1: &[f32], p2: &[f32], rng: &mut impl Rng) -> Vec<f32> {
    p1.iter().zip(p2.iter()).map(|(&a, &b)| if rng.gen::<bool>() { a } else { b }).collect()
}
