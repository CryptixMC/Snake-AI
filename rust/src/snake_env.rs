use rand::Rng;
use std::collections::HashSet;


const SENSORS: [(i32, i32); 8] =
    [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)];
const MOVES: [(i32, i32); 4] = [(-1, 0), (0, 1), (1, 0), (0, -1)];
const OPPOSITE: [usize; 4] = [2, 3, 0, 1];

pub struct SnakeEnv {
    pub grid_size:  i32,
    pub snake:      Vec<(i32, i32)>,
    pub direction:  usize,
    pub food:       (i32, i32),
    pub steps:      usize,
    pub score:      usize,
    hunger:         usize,
    hunger_limit:   usize,
    max_steps:      usize,
    food_sequence:  Vec<(i32, i32)>,
    food_idx:       usize,
}

impl SnakeEnv {
    pub fn new(grid_size: i32, food_sequence: Vec<(i32, i32)>) -> Self {
        let mid = grid_size / 2;
        let mut env = Self {
            grid_size,
            snake:         vec![(mid, mid), (mid, mid - 1), (mid, mid - 2)],
            direction:     1,
            food:          (0, 0),
            steps:         0,
            score:         0,
            hunger:        0,
            hunger_limit:  (grid_size * grid_size / 2) as usize,
            max_steps:     (grid_size * grid_size * 10) as usize,
            food_sequence,
            food_idx:      0,
        };
        env.food = env.place_food();
        env
    }

    pub fn reset(&mut self) -> Vec<f32> {
        let mid = self.grid_size / 2;
        self.snake     = vec![(mid, mid), (mid, mid - 1), (mid, mid - 2)];
        self.direction = 1;
        self.food_idx  = 0;
        self.steps     = 0;
        self.score     = 0;
        self.hunger    = 0;
        self.food      = self.place_food();
        self.get_obs()
    }

    fn place_food(&mut self) -> (i32, i32) {
        let snake_set: HashSet<_> = self.snake.iter().cloned().collect();
        let mut rng = rand::thread_rng();

        while self.food_idx < self.food_sequence.len() {
            let pos = self.food_sequence[self.food_idx];
            self.food_idx += 1;
            if !snake_set.contains(&pos) {
                return pos;
            }
        }

        loop {
            let pos = (rng.gen_range(0..self.grid_size), rng.gen_range(0..self.grid_size));
            if !snake_set.contains(&pos) {
                return pos;
            }
        }
    }

    /// Returns (obs, done).
    pub fn step(&mut self, action: usize) -> (Vec<f32>, bool) {
        if action != OPPOSITE[self.direction] {
            self.direction = action;
        }

        let (dr, dc) = MOVES[self.direction];
        let (hr, hc) = self.snake[0];
        let (nr, nc) = (hr + dr, hc + dc);

        if nr < 0 || nr >= self.grid_size || nc < 0 || nc >= self.grid_size {
            return (self.get_obs(), true);
        }

        let body: HashSet<_> = self.snake[..self.snake.len() - 1].iter().cloned().collect();
        if body.contains(&(nr, nc)) {
            return (self.get_obs(), true);
        }

        self.snake.insert(0, (nr, nc));

        if (nr, nc) == self.food {
            self.score  += 1;
            self.hunger  = 0;
            self.food    = self.place_food();
        } else {
            self.snake.pop();
            self.hunger += 1;
            if self.hunger >= self.hunger_limit {
                return (self.get_obs(), true);
            }
        }

        self.steps += 1;
        if self.steps >= self.max_steps {
            return (self.get_obs(), true);
        }

        (self.get_obs(), false)
    }

    pub fn get_obs(&self) -> Vec<f32> {
        let mut obs = vec![0f32; crate::network::IN_SIZE];
        let (hr, hc) = self.snake[0];
        let snake_set: HashSet<_> = self.snake.iter().cloned().collect();
        let (fr, fc) = self.food;

        for (i, &(dr, dc)) in SENSORS.iter().enumerate() {
            let (mut r, mut c) = (hr + dr, hc + dc);
            let mut dist = 1i32;
            let mut found_body = false;

            while r >= 0 && r < self.grid_size && c >= 0 && c < self.grid_size {
                if !found_body && snake_set.contains(&(r, c)) {
                    obs[i + 8] = 1.0 / dist as f32;
                    found_body = true;
                }
                r += dr;
                c += dc;
                dist += 1;
            }
            obs[i] = 1.0 / dist as f32;
        }

        let g = self.grid_size as f32;
        obs[16] = (hr - fr).max(0) as f32 / g;  // food is above
        obs[17] = (fr - hr).max(0) as f32 / g;  // food is below
        obs[18] = (hc - fc).max(0) as f32 / g;  // food is left
        obs[19] = (fc - hc).max(0) as f32 / g;  // food is right
        obs
    }
}
