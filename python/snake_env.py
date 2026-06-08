import numpy as np

# Cardinal + diagonal directions for 8 sensors
SENSORS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

# Absolute movement: 0=up 1=right 2=down 3=left
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]
OPPOSITES = {0: 2, 2: 0, 1: 3, 3: 1}


class SnakeEnv:
    def __init__(self, grid_size=20, food_sequence=None):
        """
        food_sequence: list of (row, col) positions used in order for food placement.
                       All snakes in a generation share the same sequence so fitness
                       is comparable. Falls back to random if the list runs out or a
                       position is occupied.
        """
        self.grid_size = grid_size
        self.food_sequence = food_sequence or []
        self._food_idx = 0
        self.reset()

    def reset(self):
        mid = self.grid_size // 2
        self.snake = [(mid, mid), (mid, mid - 1), (mid, mid - 2)]
        self.direction = 1  # facing right
        self._food_idx = 0
        self.food = self._place_food()
        self.steps = 0
        self.score = 0
        self._hunger = 0
        # Die if no food eaten within grid_size*2 steps — kills spinners fast
        self._hunger_limit = self.grid_size * self.grid_size // 2
        self._max_steps = self.grid_size * self.grid_size * 10  # absolute safety cap
        return self._get_obs()

    def _place_food(self):
        snake_set = set(self.snake)

        # Try the pre-generated sequence first
        while self._food_idx < len(self.food_sequence):
            pos = self.food_sequence[self._food_idx]
            self._food_idx += 1
            if pos not in snake_set:
                return pos

        # Random fallback (sequence exhausted or all positions occupied)
        while True:
            pos = (
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size),
            )
            if pos not in snake_set:
                return pos

    def step(self, action: int):
        if action != OPPOSITES[self.direction]:
            self.direction = action

        dr, dc = MOVES[self.direction]
        nr, nc = self.snake[0][0] + dr, self.snake[0][1] + dc

        if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
            return self._get_obs(), 0.0, True

        if (nr, nc) in set(self.snake[:-1]):
            return self._get_obs(), 0.0, True

        self.snake.insert(0, (nr, nc))

        if (nr, nc) == self.food:
            self.score += 1
            self._hunger = 0
            self.food = self._place_food()
        else:
            self.snake.pop()
            self._hunger += 1
            if self._hunger >= self._hunger_limit:
                return self._get_obs(), 0.0, True

        self.steps += 1
        if self.steps >= self._max_steps:
            return self._get_obs(), 0.0, True

        return self._get_obs(), 0.0, False

    def _get_obs(self) -> np.ndarray:
        # 20 inputs: 8 wall proximities + 8 body proximities + 4 food directions
        # Food inputs are split into non-negative components so ReLU neurons
        # can directly activate on each direction without sign detection.
        obs = np.zeros(20, dtype=np.float32)
        hr, hc = self.snake[0]
        snake_set = set(self.snake)
        fr, fc = self.food

        for i, (dr, dc) in enumerate(SENSORS):
            r, c = hr + dr, hc + dc
            dist = 1
            found_body = False

            while 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                if not found_body and (r, c) in snake_set:
                    obs[i + 8] = 1.0 / dist
                    found_body = True
                r += dr
                c += dc
                dist += 1

            obs[i] = 1.0 / dist  # wall proximity

        g = self.grid_size
        obs[16] = max(0, hr - fr) / g  # food is above  (head_row > food_row)
        obs[17] = max(0, fr - hr) / g  # food is below
        obs[18] = max(0, hc - fc) / g  # food is left
        obs[19] = max(0, fc - hc) / g  # food is right

        return obs
