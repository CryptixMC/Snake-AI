import argparse
import time
import torch
from ga import GeneticAlgorithm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=int, default=500)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--grid", type=int, default=20)
    parser.add_argument("--mutation-std", type=float, default=0.3)
    parser.add_argument("--mutation-rate", type=float, default=0.05,
                        help="Fraction of weights perturbed per offspring (0.05=5%)")
    parser.add_argument("--selection-pressure", type=float, default=3.0,
                        help="How much more likely best is vs worst (1.0=random, 3.0=moderate)")
    parser.add_argument("--visualize-every", type=int, default=10,
                        help="Show top snakes every N generations (0=off)")
    parser.add_argument("--viz-snakes", type=int, default=10,
                        help="Number of snakes to show simultaneously during visualization")
    parser.add_argument("--resume", type=str, default=None, metavar="FILE",
                        help="Seed the population from a .pt checkpoint and continue training")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Population: {args.population}  Generations: {args.generations}  Grid: {args.grid}x{args.grid}\n")

    ga = GeneticAlgorithm(
        population_size=args.population,
        grid_size=args.grid,
        device=DEVICE,
        mutation_std=args.mutation_std,
        mutation_rate=args.mutation_rate,
        selection_pressure=args.selection_pressure,
    )

    if args.resume:
        seed = torch.load(args.resume, map_location=DEVICE, weights_only=True)
        ga.params.copy_(seed.unsqueeze(0).expand_as(ga.params))
        print(f"Seeded population from {args.resume}\n")

    viz = None
    if args.visualize_every > 0:
        try:
            import pygame  # noqa: F401
            from visualize import Visualizer
            viz = Visualizer(grid_size=args.grid)
        except ImportError:
            print("pygame not available — running headless")

    best_ever = 0.0
    best_params = None

    for gen in range(1, args.generations + 1):
        t0 = time.perf_counter()
        stats = ga.step()
        elapsed = time.perf_counter() - t0

        if stats["best_fitness"] > best_ever:
            best_ever = stats["best_fitness"]
            best_params = ga.top_n_params(1)[0].clone()

        print(
            f"gen {gen:4d}  "
            f"score {stats['best_score']:3d}  "
            f"best_ever {int(best_ever)//1000:3d}  "
            f"avg_fit {stats['avg_fitness']:7.0f}  "
            f"({elapsed:.1f}s)"
        )

        if viz and args.visualize_every > 0 and gen % args.visualize_every == 0:
            n = min(args.viz_snakes, ga.N)
            result = viz.play_top_n(ga.top_n_params(n), DEVICE, generation=gen)
            if result == -1:  # user closed window
                break

    if viz:
        viz.close()

    if best_params is not None:
        torch.save(best_params, "best_snake.pt")
        print(f"\nSaved best individual to best_snake.pt  (fitness {best_ever:.0f})")


if __name__ == "__main__":
    main()
