use macroquad::prelude::*;

use crate::network::{forward_with_activations, IN_SIZE, H_SIZE, OUT_SIZE};
use crate::snake_env::SnakeEnv;

// ── Layout constants ─────────────────────────────────────────────────────────
pub const CELL:    f32 = 28.0;
pub const GRID:    i32 = 20;
pub const GAME_W:  f32 = CELL * GRID as f32;   // 560
pub const GAME_H:  f32 = GAME_W;               // 560
pub const HUD_H:   f32 = 50.0;
pub const PANEL_X: f32 = GAME_W + 10.0;        // 570
pub const WIN_W:   f32 = PANEL_X + 370.0;      // 940
pub const WIN_H:   f32 = GAME_H + HUD_H;       // 610

const INP_X:  f32 = PANEL_X + 50.0;
const HID_X:  f32 = PANEL_X + 185.0;
const OUT_X:  f32 = PANEL_X + 310.0;
const LOUT_X: f32 = PANEL_X + 323.0;

const R_INP: f32 = 5.0;
const R_HID: f32 = 3.0;  // smaller — 64 nodes need tighter packing
const R_OUT: f32 = 9.0;

const SENSOR_DIRS: [&str; 8] = ["N","NE","E","SE","S","SW","W","NW"];
const OUT_LABELS:  [&str; 4] = ["^", ">", "v", "<"];

// group (name, dark rgb, bright rgb)
const WALL_DARK:  (f32,f32,f32) = (0.14, 0.06, 0.06);
const WALL_BRIGHT:(f32,f32,f32) = (0.90, 0.29, 0.29);
const BODY_DARK:  (f32,f32,f32) = (0.14, 0.11, 0.04);
const BODY_BRIGHT:(f32,f32,f32) = (0.90, 0.67, 0.16);
const FOOD_DARK:  (f32,f32,f32) = (0.04, 0.08, 0.16);
const FOOD_BRIGHT:(f32,f32,f32) = (0.24, 0.63, 1.00);
const HID_DARK:   (f32,f32,f32) = (0.09, 0.09, 0.09);
const HID_BRIGHT: (f32,f32,f32) = (1.00, 0.78, 0.20);

fn lerp_col(a: (f32,f32,f32), b: (f32,f32,f32), t: f32) -> Color {
    let t = t.clamp(0.0, 1.0);
    Color::new(a.0+(b.0-a.0)*t, a.1+(b.1-a.1)*t, a.2+(b.2-a.2)*t, 1.0)
}

fn rank_color(rank: usize, n: usize) -> Color {
    let t = rank as f32 / (n.saturating_sub(1).max(1)) as f32;
    let h = 0.38f32;
    let s = (1.0 - t * 0.85).max(0.12);
    let v = (1.0 - t * 0.68).max(0.28);
    hsv(h, s, v)
}

// standard HSV → RGBA (macroquad helper)
fn hsv(h: f32, s: f32, v: f32) -> Color {
    let i = (h * 6.0).floor() as i32;
    let f = h * 6.0 - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    let (r, g, b) = match i % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    Color::new(r, g, b, 1.0)
}

fn make_ys(n: usize, top: f32, bottom: f32) -> Vec<f32> {
    if n == 1 { return vec![(top + bottom) / 2.0]; }
    (0..n).map(|i| top + i as f32 * (bottom - top) / (n - 1) as f32).collect()
}

pub struct Visualizer {
    pub grid_size: i32,
    inp_ys: Vec<f32>,
    hid_ys: Vec<f32>,
    out_ys: Vec<f32>,
}

impl Visualizer {
    pub fn new(grid_size: i32) -> Self {
        Self {
            grid_size,
            inp_ys: make_ys(IN_SIZE, 18.0, 542.0),
            hid_ys: make_ys(H_SIZE,  18.0, 542.0),
            out_ys: make_ys(OUT_SIZE, 115.0, 495.0),
        }
    }

    /// Play n snakes simultaneously. Returns best score, or -1 if user quits.
    pub async fn play_top_n(
        &self,
        params_list: &[Vec<f32>],
        generation:  usize,
        fps:         f64,
    ) -> i32 {
        let n       = params_list.len();
        let mut envs: Vec<SnakeEnv> = (0..n).map(|_| SnakeEnv::new(self.grid_size, vec![])).collect();
        let mut obs:  Vec<Vec<f32>> = envs.iter_mut().map(|e| e.reset()).collect();
        let mut alive = vec![true; n];
        let colors: Vec<Color> = (0..n).map(|i| rank_color(i, n)).collect();
        let mut best_acts = None;

        let step_dt = 1.0 / fps;
        let mut last_step = get_time();

        loop {
            // ── Input ──
            if is_key_pressed(KeyCode::Q) || is_key_pressed(KeyCode::Escape) {
                return -1;
            }

            // ── Step snakes at target fps ──
            let now = get_time();
            if now - last_step >= step_dt {
                last_step = now;
                for i in 0..n {
                    if !alive[i] { continue; }
                    let acts = forward_with_activations(&params_list[i], &obs[i]);
                    let (new_obs, done) = envs[i].step(acts.action);
                    obs[i] = new_obs;
                    if i == 0 || best_acts.is_none() {
                        // track best-alive for NN display (first alive = rank 0)
                        if alive[i] { best_acts = Some(acts); }
                    }
                    if done { alive[i] = false; }
                }
                // find best still alive for NN display
                best_acts = None;
                for i in 0..n {
                    if alive[i] {
                        best_acts = Some(forward_with_activations(&params_list[i], &obs[i]));
                        break;
                    }
                }
            }

            // ── Draw ──
            clear_background(Color::from_rgba(15, 15, 15, 255));
            self.draw_game(&envs, &alive, &colors);
            if let Some(ref acts) = best_acts {
                self.draw_panel(acts);
            }
            self.draw_hud(&envs, &alive, &colors, generation);

            next_frame().await;

            if alive.iter().all(|&a| !a) {
                // brief pause before returning
                let end = get_time() + 0.7;
                while get_time() < end { next_frame().await; }
                let best = envs.iter().map(|e| e.score).max().unwrap_or(0);
                return best as i32;
            }
        }
    }

    // ── Game panel ───────────────────────────────────────────────────────────

    fn draw_game(&self, envs: &[SnakeEnv], alive: &[bool], colors: &[Color]) {
        let c = CELL;

        // Grid
        let grid_col = Color::from_rgba(25, 25, 25, 255);
        for i in 0..=(self.grid_size as usize) {
            let x = i as f32 * c;
            let y = i as f32 * c;
            draw_line(x, 0.0, x, GAME_H, 1.0, grid_col);
            draw_line(0.0, y, GAME_W, y, 1.0, grid_col);
        }

        // Food — draw worst-first so best food is on top
        for i in (0..envs.len()).rev() {
            if !alive[i] { continue; }
            let (fr, fc) = envs[i].food;
            let col = colors[i];
            let food_col = Color::new(col.r * 0.5, col.g * 0.5, col.b * 0.5, 1.0);
            draw_circle(fc as f32 * c + c / 2.0, fr as f32 * c + c / 2.0, 4.0, food_col);
        }

        // Snakes — worst first so best renders on top
        for i in (0..envs.len()).rev() {
            if !alive[i] { continue; }
            let col = colors[i];
            let head = Color::new(
                (col.r * 1.45).min(1.0),
                (col.g * 1.45).min(1.0),
                (col.b * 1.45).min(1.0),
                1.0,
            );
            for (idx, &(r, col_pos)) in envs[i].snake.iter().enumerate() {
                let clr = if idx == 0 { head } else { col };
                draw_rectangle(col_pos as f32 * c + 1.0, r as f32 * c + 1.0, c - 2.0, c - 2.0, clr);
            }
        }
    }

    // ── NN panel ─────────────────────────────────────────────────────────────

    fn draw_panel(&self, acts: &crate::network::ForwardResult) {
        // Panel background
        draw_rectangle(PANEL_X, 0.0, WIN_W - PANEL_X, WIN_H, Color::from_rgba(20, 20, 20, 255));
        draw_line(PANEL_X - 1.0, 0.0, PANEL_X - 1.0, WIN_H, 1.0, Color::from_rgba(50, 50, 50, 255));

        draw_text("NEURAL NETWORK", PANEL_X + 100.0, 14.0, 14.0, Color::from_rgba(70, 70, 80, 255));
        let dim = Color::from_rgba(70, 70, 80, 255);
        draw_text("INPUT",  INP_X - 18.0, GAME_H - 4.0, 12.0, dim);
        draw_text("HIDDEN", HID_X - 22.0, GAME_H - 4.0, 12.0, dim);
        draw_text("OUTPUT", OUT_X - 24.0, GAME_H - 4.0, 12.0, dim);

        self.draw_connections(acts);
        self.draw_neurons(acts);
    }

    fn draw_connections(&self, acts: &crate::network::ForwardResult) {
        let inp = &acts.inputs;
        let hid = &acts.hidden;
        let w1  = &acts.w1;
        let w2  = &acts.w2;

        for i in 0..IN_SIZE {
            for j in 0..H_SIZE {
                let contrib = (w1[i][j] * inp[i]).abs();
                let alpha = (contrib * 0.7).min(0.82);
                if alpha < 0.03 { continue; }
                let (r, g, b) = if w1[i][j] > 0.0 { (0.31, 0.55, 1.0) } else { (1.0, 0.31, 0.31) };
                draw_line(INP_X, self.inp_ys[i], HID_X, self.hid_ys[j], 1.0,
                    Color::new(r, g, b, alpha));
            }
        }

        for j in 0..H_SIZE {
            for k in 0..OUT_SIZE {
                let contrib = (w2[j][k] * hid[j]).abs();
                let alpha = (contrib * 0.7).min(0.82);
                if alpha < 0.03 { continue; }
                let (r, g, b) = if w2[j][k] > 0.0 { (0.31, 0.55, 1.0) } else { (1.0, 0.31, 0.31) };
                draw_line(HID_X, self.hid_ys[j], OUT_X, self.out_ys[k], 1.0,
                    Color::new(r, g, b, alpha));
            }
        }
    }

    fn draw_neurons(&self, acts: &crate::network::ForwardResult) {
        let inp    = &acts.inputs;
        let hid    = &acts.hidden;
        let out    = &acts.output;
        let action = acts.action;

        let hid_max = hid.iter().cloned().fold(1e-6f32, f32::max);

        // Softmax for output display
        let out_max = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = out.iter().map(|&v| (v - out_max).exp()).collect();
        let exp_sum: f32  = exp.iter().sum();
        let softmax: Vec<f32> = exp.iter().map(|&e| e / exp_sum).collect();

        let dim_text = Color::from_rgba(65, 65, 75, 255);

        // Input neurons
        for i in 0..IN_SIZE {
            let clr = if i < 8 {
                lerp_col(WALL_DARK, WALL_BRIGHT, inp[i])
            } else if i < 16 {
                lerp_col(BODY_DARK, BODY_BRIGHT, inp[i])
            } else {
                lerp_col(FOOD_DARK, FOOD_BRIGHT, inp[i])  // already non-negative
            };
            draw_circle(INP_X, self.inp_ys[i], R_INP, clr);

            let food_lbls = ["^", "v", "<", ">"];
            let lbl = if i < 16 { SENSOR_DIRS[i % 8] } else { food_lbls[i - 16] };
            draw_text(lbl, INP_X - 14.0, self.inp_ys[i] + 4.0, 11.0, dim_text);
        }

        // Group labels
        let wall_mid = (self.inp_ys[0] + self.inp_ys[7]) / 2.0;
        let body_mid = (self.inp_ys[8] + self.inp_ys[15]) / 2.0;
        let food_mid = (self.inp_ys[16] + self.inp_ys[19]) / 2.0;
        draw_text("WALL", PANEL_X + 2.0, wall_mid + 4.0, 11.0, Color::from_rgba(230, 75, 75, 255));
        draw_text("BODY", PANEL_X + 2.0, body_mid + 4.0, 11.0, Color::from_rgba(230, 170, 40, 255));
        draw_text("FOOD", PANEL_X + 2.0, food_mid + 4.0, 11.0, Color::from_rgba(60, 160, 255, 255));

        // Hidden neurons — draw every other one to avoid overdraw at 64 nodes
        for j in 0..H_SIZE {
            let t = hid[j] / hid_max;
            draw_circle(HID_X, self.hid_ys[j], R_HID, lerp_col(HID_DARK, HID_BRIGHT, t));
        }
        let _h = H_SIZE; // suppress unused warning

        // Output neurons
        for k in 0..OUT_SIZE {
            let clr = if k == action {
                Color::from_rgba(30, 230, 90, 255)
            } else {
                let v = (softmax[k] * 160.0) as u8;
                Color::from_rgba(v, v, v, 255)
            };
            draw_circle(OUT_X, self.out_ys[k], R_OUT, clr);
            draw_text(OUT_LABELS[k], LOUT_X, self.out_ys[k] + 5.0, 16.0, WHITE);
        }
    }

    // ── HUD ──────────────────────────────────────────────────────────────────

    fn draw_hud(&self, envs: &[SnakeEnv], alive: &[bool], colors: &[Color], generation: usize) {
        let best_score = envs.iter().map(|e| e.score).max().unwrap_or(0);
        let n_alive    = alive.iter().filter(|&&a| a).count();
        let n          = envs.len();

        let txt = format!("gen {}   best score {}   alive {}/{}", generation, best_score, n_alive, n);
        draw_text(&txt, 8.0, GAME_H + 22.0, 16.0, Color::from_rgba(200, 200, 200, 255));

        // Color swatch legend
        let max_swatches = n.min(40);
        let mut lx = 8.0f32;
        for i in 0..max_swatches {
            draw_rectangle(lx, GAME_H + 36.0, 10.0, 6.0, colors[i]);
            lx += 12.0;
        }
        let lbl = if n <= 40 { "best->worst".to_string() } else { format!("best->worst ({n})") };
        draw_text(&lbl, lx + 4.0, GAME_H + 44.0, 11.0, Color::from_rgba(70, 70, 80, 255));
    }
}
