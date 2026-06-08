import colorsys
import numpy as np
import pygame
import torch
from snake_env import SnakeEnv
from network import forward_with_activations

# ── Layout ──────────────────────────────────────────────────────────────────
CELL   = 28
GRID   = 20
GAME_W = GRID * CELL   # 560
GAME_H = GRID * CELL   # 560
HUD_H  = 50
GAP    = 10

PANEL_X = GAME_W + GAP  # 570
PANEL_W = 370
WIN_W   = PANEL_X + PANEL_W  # 940
WIN_H   = GAME_H + HUD_H     # 610

# NN node x positions (absolute)
INP_X  = PANEL_X + 50
HID_X  = PANEL_X + 185
OUT_X  = PANEL_X + 310
LOUT_X = PANEL_X + 327

# Node radii
R_INP = 5
R_HID = 3   # smaller — 64 nodes need tighter packing
R_OUT = 9

# Vertical node positions
def _ys(n, top=18, bottom=GAME_H - 18):
    if n == 1:
        return [(top + bottom) // 2]
    return [int(top + i * (bottom - top) / (n - 1)) for i in range(n)]

INP_YS = _ys(20)
HID_YS = _ys(64)
OUT_YS = _ys(4, top=115, bottom=495)

SENSOR_DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
OUT_LABELS  = ["↑", "→", "↓", "←"]

GROUPS = [
    ("WALL", (35, 15, 15), (230,  75,  75)),
    ("BODY", (35, 28, 10), (230, 170,  40)),
]
FOOD_DARK   = (10, 20, 40)
FOOD_BRIGHT = (60, 160, 255)
FOOD_LABELS = ["↑", "↓", "←", "→"]

HID_DARK   = (22, 22, 22)
HID_BRIGHT = (255, 200, 50)

BG       = (15,  15,  15)
GRID_C   = (25,  25,  25)
TEXT_C   = (200, 200, 200)
PANEL_BG = (20,  20,  20)
SEP_C    = (50,  50,  50)
DIM_C    = (70,  70,  80)
ACT_POS  = ( 80, 140, 255)
ACT_NEG  = (255,  80,  80)


def _lerp(a, b, t):
    t = max(0.0, min(1.0, float(t)))
    return (int(a[0] + (b[0]-a[0])*t), int(a[1] + (b[1]-a[1])*t), int(a[2] + (b[2]-a[2])*t))


def _rank_color(rank: int, n: int = 10) -> tuple:
    """Bright saturated green (rank 0) → dim desaturated (rank n-1)."""
    t = rank / max(n - 1, 1)
    h = 0.38  # green-cyan hue
    s = max(0.12, 1.0 - t * 0.85)
    v = max(0.28, 1.0 - t * 0.68)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


class Visualizer:
    def __init__(self, grid_size: int = GRID, cell: int = CELL):
        self.grid_size = grid_size
        self.cell = cell

        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Snake AI")
        self.font_md = pygame.font.SysFont("monospace", 15)
        self.font_sm = pygame.font.SysFont("monospace", 11)
        self.clock   = pygame.time.Clock()

        self.conn_surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)

    def play_top_n(
        self,
        params_list: list,
        device: torch.device,
        generation: int,
        fps: int = 15,
    ) -> int:
        """
        Play n snake episodes simultaneously (ordered best→worst).
        Snakes are overlaid on one grid; best = most saturated color.
        NN panel shows the best alive snake. Returns best score, -1 on quit.
        """
        n = len(params_list)
        envs    = [SnakeEnv(self.grid_size) for _ in range(n)]
        obs_arr = [env.reset() for env in envs]
        params  = [p.to(device) for p in params_list]
        alive   = [True] * n
        colors  = [_rank_color(i, n) for i in range(n)]
        best_acts = None

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return -1
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    return -1

            step_acts = []
            for i in range(n):
                if not alive[i]:
                    step_acts.append(None)
                    continue
                obs_t = torch.tensor(obs_arr[i], device=device)
                with torch.no_grad():
                    acts = forward_with_activations(params[i], obs_t)
                step_acts.append(acts)
                obs_arr[i], _, done = envs[i].step(acts["action"])
                if done:
                    alive[i] = False

            # NN display tracks the highest-ranked snake still alive
            for i in range(n):
                if alive[i]:
                    best_acts = step_acts[i]
                    break

            self.screen.fill(BG)
            self._draw_game_multi(envs, alive, colors)
            if best_acts is not None:
                self._draw_panel(best_acts)
            self._draw_hud_multi(envs, alive, colors, generation)
            pygame.display.flip()
            self.clock.tick(fps)

            if not any(alive):
                pygame.time.wait(700)
                return max(env.score for env in envs)

    # ── Game rendering ───────────────────────────────────────────────────────

    def _draw_game_multi(self, envs, alive, colors):
        c = self.cell

        for i in range(self.grid_size + 1):
            pygame.draw.line(self.screen, GRID_C, (i*c, 0), (i*c, GAME_H))
            pygame.draw.line(self.screen, GRID_C, (0, i*c), (GAME_W, i*c))

        # Food dots — small, tinted to match each snake
        for i, env in enumerate(envs):
            if not alive[i]:
                continue
            fr, fc = env.food
            r, g, b = colors[i]
            pygame.draw.circle(
                self.screen, (r//2, g//2, b//2),
                (fc*c + c//2, fr*c + c//2), 4,
            )

        # Snakes — draw worst first so best renders on top
        for i in range(len(envs) - 1, -1, -1):
            if not alive[i]:
                continue
            clr = colors[i]
            head = tuple(min(255, int(x * 1.45)) for x in clr)
            for idx, (r, col) in enumerate(envs[i].snake):
                pygame.draw.rect(
                    self.screen, head if idx == 0 else clr,
                    (col*c + 1, r*c + 1, c - 2, c - 2),
                )

    # ── NN panel ─────────────────────────────────────────────────────────────

    def _draw_panel(self, acts: dict):
        pygame.draw.rect(self.screen, PANEL_BG, (PANEL_X, 0, PANEL_W, GAME_H + HUD_H))
        pygame.draw.line(self.screen, SEP_C, (PANEL_X - 1, 0), (PANEL_X - 1, WIN_H))

        self.screen.blit(
            self.font_sm.render("NEURAL  NETWORK", True, DIM_C),
            (PANEL_X + 110, 4),
        )
        for text, x in [("INPUT", INP_X-18), ("HIDDEN", HID_X-20), ("OUTPUT", OUT_X-22)]:
            self.screen.blit(self.font_sm.render(text, True, DIM_C), (x, GAME_H - 14))

        self._draw_connections(acts)
        self._draw_neurons(acts)

    def _draw_connections(self, acts: dict):
        self.conn_surf.fill((0, 0, 0, 0))

        inp = acts["inputs"]
        hid = acts["hidden"]
        w1  = acts["w1"]
        w2  = acts["w2"]

        for i in range(20):
            for j in range(16):
                contrib = abs(float(w1[i, j]) * float(inp[i]))
                alpha = min(210, int(contrib * 700))
                if alpha < 8:
                    continue
                clr = (*ACT_POS, alpha) if w1[i, j] > 0 else (*ACT_NEG, alpha)
                pygame.draw.line(self.conn_surf, clr, (INP_X, INP_YS[i]), (HID_X, HID_YS[j]))

        for j in range(16):
            for k in range(4):
                contrib = abs(float(w2[j, k]) * float(hid[j]))
                alpha = min(210, int(contrib * 700))
                if alpha < 8:
                    continue
                clr = (*ACT_POS, alpha) if w2[j, k] > 0 else (*ACT_NEG, alpha)
                pygame.draw.line(self.conn_surf, clr, (HID_X, HID_YS[j]), (OUT_X, OUT_YS[k]))

        self.screen.blit(self.conn_surf, (0, 0))

    def _draw_neurons(self, acts: dict):
        inp    = acts["inputs"]
        hid    = acts["hidden"]
        out    = acts["output"]
        action = acts["action"]

        hid_max = max(float(hid.max()), 1e-6)
        exp_o   = np.exp(out - out.max())
        softmax = exp_o / exp_o.sum()

        for i in range(16):
            g = i // 8
            _, dark, bright = GROUPS[g]
            pygame.draw.circle(self.screen, _lerp(dark, bright, float(inp[i])),
                               (INP_X, INP_YS[i]), R_INP)
            self.screen.blit(
                self.font_sm.render(SENSOR_DIRS[i % 8], True, (65, 65, 75)),
                (INP_X - 24, INP_YS[i] - 5),
            )

        # Food direction nodes 16-19 (all non-negative)
        for j in range(4):
            fi = 16 + j
            pygame.draw.circle(self.screen, _lerp(FOOD_DARK, FOOD_BRIGHT, float(inp[fi])),
                               (INP_X, INP_YS[fi]), R_INP)
            self.screen.blit(
                self.font_sm.render(FOOD_LABELS[j], True, (65, 65, 75)),
                (INP_X - 14, INP_YS[fi] - 5),
            )

        for g, (name, _, bright) in enumerate(GROUPS):
            mid_y = (INP_YS[g*8] + INP_YS[g*8+7]) // 2
            self.screen.blit(self.font_sm.render(name, True, bright), (PANEL_X+3, mid_y-5))
        food_mid = (INP_YS[16] + INP_YS[19]) // 2
        self.screen.blit(self.font_sm.render("FOOD", True, FOOD_BRIGHT), (PANEL_X+3, food_mid-5))

        for j in range(64):
            t = float(hid[j]) / hid_max
            pygame.draw.circle(self.screen, _lerp(HID_DARK, HID_BRIGHT, t),
                               (HID_X, HID_YS[j]), R_HID)

        for k in range(4):
            clr = (30, 230, 90) if k == action else (int(softmax[k]*160),)*3
            pygame.draw.circle(self.screen, clr, (OUT_X, OUT_YS[k]), R_OUT)
            self.screen.blit(self.font_md.render(OUT_LABELS[k], True, TEXT_C),
                             (LOUT_X, OUT_YS[k]-8))

    # ── HUD ──────────────────────────────────────────────────────────────────

    def _draw_hud_multi(self, envs, alive, colors, generation):
        best_score = max(env.score for env in envs)
        n_alive    = sum(alive)
        n          = len(envs)

        hud = self.font_md.render(
            f"gen {generation}   best score {best_score}   alive {n_alive}/{n}",
            True, TEXT_C,
        )
        self.screen.blit(hud, (8, GAME_H + 16))

        # Color legend: up to 40 swatches before the label would overflow
        lx = 8
        max_swatches = min(len(colors), 40)
        for clr in colors[:max_swatches]:
            pygame.draw.rect(self.screen, clr, (lx, GAME_H + 36, 10, 6))
            lx += 12
        label = "best→worst" if len(colors) <= 40 else f"best→worst ({len(colors)} snakes)"
        self.screen.blit(self.font_sm.render(label, True, DIM_C), (lx + 4, GAME_H + 33))

        # Connection legend
        lx2 = PANEL_X + 8
        ly  = GAME_H + 12
        for clr, txt in [(ACT_POS, "+weight"), (ACT_NEG, "-weight")]:
            pygame.draw.line(self.screen, clr, (lx2, ly+5), (lx2+18, ly+5), 2)
            self.screen.blit(self.font_sm.render(txt, True, DIM_C), (lx2+22, ly))
            lx2 += 90

    def close(self):
        pygame.quit()
