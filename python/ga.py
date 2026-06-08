import numpy as np
import torch
from snake_env import SnakeEnv
from network import forward_batch, random_population, PARAM_COUNT


def evaluate_population(
    params: torch.Tensor,
    grid_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    Run all N snake episodes in parallel using batched GPU inference.
    All snakes share the same food sequence so fitness scores are comparable.
    """
    N = params.shape[0]
    max_dist = (grid_size - 1) * 2  # max possible manhattan distance on grid

    food_seq = [
        (int(np.random.randint(0, grid_size)), int(np.random.randint(0, grid_size)))
        for _ in range(500)
    ]
    envs = [SnakeEnv(grid_size, food_sequence=food_seq) for _ in range(N)]

    obs_np = np.stack([env.reset() for env in envs])  # (N, 18)
    alive  = np.ones(N, dtype=bool)

    global_max = grid_size * grid_size * 10

    for _ in range(global_max):
        if not alive.any():
            break

        obs_t = torch.from_numpy(obs_np).to(device)
        with torch.no_grad():
            actions = forward_batch(params, obs_t).argmax(dim=1).cpu().numpy()

        for i in range(N):
            if not alive[i]:
                continue
            obs_np[i], _, done = envs[i].step(int(actions[i]))
            if done:
                alive[i] = False

    fitness = np.zeros(N, dtype=np.float32)
    for i, env in enumerate(envs):
        hr, hc = env.snake[0]
        fr, fc = env.food
        dist_bonus = max_dist - (abs(hr - fr) + abs(hc - fc))
        if env.score > 0:
            fitness[i] = env.score * 1000 - env.steps * 0.1 + dist_bonus
        else:
            fitness[i] = float(dist_bonus)

    return fitness


class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int = 500,
        grid_size: int = 20,
        device: torch.device = torch.device("cpu"),
        elite_frac: float = 0.1,
        selection_pressure: float = 3.0,
        mutation_std: float = 0.3,
        mutation_rate: float = 0.05,
        crossover_rate: float = 0.5,
    ):
        self.N                  = population_size
        self.grid_size          = grid_size
        self.device             = device
        self.elite_n            = max(1, int(population_size * elite_frac))
        self.selection_pressure = selection_pressure
        self.mutation_std       = mutation_std
        self.mutation_rate      = mutation_rate
        self.crossover_rate     = crossover_rate

        self._rank_probs: np.ndarray = self._build_rank_probs(population_size, selection_pressure)

        self.params   = random_population(population_size, device)
        self.fitness  = np.zeros(population_size, dtype=np.float32)
        self.generation = 0
        self._ranked_params: list = []

    def step(self) -> dict:
        """Run one generation: evaluate → select → reproduce."""
        self.fitness = evaluate_population(self.params, self.grid_size, self.device)
        self.generation += 1

        ranked = np.argsort(self.fitness)[::-1]
        self._ranked_params = [self.params[int(idx)].clone() for idx in ranked]

        new_params = self.params.clone()

        for i, idx in enumerate(ranked[: self.elite_n]):
            new_params[i] = self.params[idx]

        for i in range(self.elite_n, self.N):
            if np.random.rand() < self.crossover_rate:
                p1 = self.params[self._rank_select(ranked)]
                p2 = self.params[self._rank_select(ranked)]
                child = self._crossover(p1, p2)
            else:
                child = self.params[self._rank_select(ranked)].clone()
            self._mutate(child)
            new_params[i] = child

        self.params = new_params

        return {
            "generation":    self.generation,
            "best_fitness":  float(self.fitness[ranked[0]]),
            "avg_fitness":   float(self.fitness.mean()),
            "best_score":    int(self.fitness[ranked[0]]) // 1000,
        }

    @staticmethod
    def _build_rank_probs(n: int, selection_pressure: float) -> np.ndarray:
        weights = np.linspace(selection_pressure, 1.0, n)
        return weights / weights.sum()

    def _rank_select(self, ranked: np.ndarray) -> int:
        rank_pos = np.random.choice(self.N, p=self._rank_probs)
        return int(ranked[rank_pos])

    def _crossover(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        mask = torch.rand(PARAM_COUNT, device=self.device) < 0.5
        return torch.where(mask, p1, p2)

    def _mutate(self, individual: torch.Tensor):
        mask_fine = torch.rand(PARAM_COUNT, device=self.device) < self.mutation_rate
        noise_fine = torch.randn(PARAM_COUNT, device=self.device) * self.mutation_std
        mask_large = torch.rand(PARAM_COUNT, device=self.device) < 0.02
        noise_large = torch.randn(PARAM_COUNT, device=self.device) * (self.mutation_std * 5)
        individual.add_(mask_fine * noise_fine + mask_large * noise_large)

    def top_n_params(self, n: int) -> list:
        return self._ranked_params[:n]
