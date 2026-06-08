mod ga;
mod gpu;
mod network;
mod snake_env;
mod visualize;

use std::time::Instant;
use clap::{Parser, Subcommand};
use macroquad::prelude::*;

use ga::GeneticAlgorithm;
use visualize::{Visualizer, WIN_W, WIN_H};

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "snake-ai", about = "Snake neuroevolution — Rust")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train from scratch or resume from a checkpoint
    Train(TrainArgs),
    /// Watch a saved checkpoint play
    Watch(WatchArgs),
}

#[derive(Parser)]
struct TrainArgs {
    #[arg(long, default_value_t = 500)]
    population: usize,
    #[arg(long, default_value_t = 200)]
    generations: usize,
    #[arg(long, default_value_t = 20)]
    grid: i32,
    #[arg(long, default_value_t = 0.3)]
    mutation_std: f32,
    #[arg(long, default_value_t = 0.05)]
    mutation_rate: f32,
    #[arg(long, default_value_t = 3.0)]
    selection_pressure: f32,
    #[arg(long, default_value_t = 10, help = "Visualize every N gens (0=off)")]
    visualize_every: usize,
    #[arg(long, default_value_t = 10)]
    viz_snakes: usize,
    #[arg(long, default_value_t = 15.0)]
    fps: f64,
    #[arg(long, help = "Resume training from a .bin checkpoint")]
    resume: Option<String>,
    #[arg(long, default_value = "best_snake.bin")]
    output: String,
    #[arg(long, help = "Force CPU evaluation (rayon), skip GPU init")]
    cpu: bool,
}

#[derive(Parser)]
struct WatchArgs {
    checkpoint: String,
    #[arg(long, default_value_t = 15.0)]
    fps: f64,
    #[arg(long, default_value_t = 20)]
    grid: i32,
}

// ── Window config ─────────────────────────────────────────────────────────────

fn window_conf() -> Conf {
    Conf {
        window_title: "Snake AI — Rust".to_owned(),
        window_width:  WIN_W as i32,
        window_height: WIN_H as i32,
        ..Default::default()
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[macroquad::main(window_conf)]
async fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Train(args) => run_train(args).await,
        Command::Watch(args) => run_watch(args).await,
    }
}

// ── Train ────────────────────────────────────────────────────────────────────

async fn run_train(args: TrainArgs) {
    let viz = Visualizer::new(args.grid);

    let gpu = if args.cpu {
        println!("CPU mode (rayon)");
        None
    } else {
        println!("Initialising GPU...");
        let g = gpu::GpuInference::new(args.population).await;
        if g.is_none() { println!("  GPU unavailable — falling back to CPU (rayon)"); }
        g
    };

    let mut ga = GeneticAlgorithm::new(
        args.population,
        args.grid,
        0.1,
        args.selection_pressure,
        args.mutation_std,
        args.mutation_rate,
        0.5,
        gpu,
    );

    if let Some(path) = &args.resume {
        match load_params(path) {
            Ok(seed) => {
                for p in ga.params.iter_mut() { *p = seed.clone(); }
                println!("Resumed from {path}");
            }
            Err(e) => eprintln!("Failed to load {path}: {e}"),
        }
    }

    println!("Snake AI — Rust");
    println!("Population {}  Grid {}x{}  Generations {}", args.population, args.grid, args.grid, args.generations);
    println!("{}", "-".repeat(72));

    let mut best_ever = 0.0f32;
    let mut best_params: Vec<f32> = vec![];

    for gen in 1..=args.generations {
        // Window stays responsive — drain events once per generation
        next_frame().await;
        if is_key_pressed(KeyCode::Q) || is_key_pressed(KeyCode::Escape) {
            break;
        }

        let t0    = Instant::now();
        let stats = ga.step();
        let secs  = t0.elapsed().as_secs_f32();

        if stats.best_fitness > best_ever {
            best_ever   = stats.best_fitness;
            best_params = ga.top_n_params(1).remove(0);
        }

        println!(
            "gen {:4}  score {:3}  best_ever {:3}  avg_fit {:7.0}  ({:.1}s)",
            stats.generation, stats.best_score,
            (best_ever as usize) / 1000,
            stats.avg_fitness, secs
        );

        if args.visualize_every > 0 && gen % args.visualize_every == 0 {
            let n      = args.viz_snakes.min(ga.population_size);
            let result = viz.play_top_n(&ga.top_n_params(n), gen, args.fps).await;
            if result < 0 { break; }
        }
    }

    if !best_params.is_empty() {
        if let Err(e) = save_params(&best_params, &args.output) {
            eprintln!("Failed to save: {e}");
        } else {
            println!("\nSaved best to {}  (fitness {:.0})", args.output, best_ever);
        }
    }
}

// ── Watch ────────────────────────────────────────────────────────────────────

async fn run_watch(args: WatchArgs) {
    let params = match load_params(&args.checkpoint) {
        Ok(p)  => p,
        Err(e) => { eprintln!("Cannot load {}: {e}", args.checkpoint); return; }
    };

    println!("Watching {}  |  press Q/Esc to quit", args.checkpoint);

    let viz = Visualizer::new(args.grid);
    let mut episode = 0usize;

    loop {
        episode += 1;
        let score = viz.play_top_n(&[params.clone()], episode, args.fps).await;
        if score < 0 { break; }
        println!("episode {:4}  score {}", episode, score);
    }
}

// ── Checkpoint I/O ────────────────────────────────────────────────────────────

fn save_params(params: &[f32], path: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    for v in params {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn load_params(path: &str) -> std::io::Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}
