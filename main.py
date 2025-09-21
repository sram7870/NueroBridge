#!/usr/bin/env python3
r"""
friend_hybrid.py — Advanced neuron-controlled Doom (Stage 1+2, custom room)

This script demonstrates a highly advanced neuron-controlled agent for ViZDoom.
The agent navigates a custom room to reach a green armor target using:

Here is where we brainstormed additions:
https://docs.google.com/document/d/1MMW6kSfGNuMMAuROP8jQACMiVK1O4hXNJn7eNtafelo/edit?usp=sharing
"""

# ─────────────────────────────────────────────────────
# Standard Libraries
# ─────────────────────────────────────────────────────
import os
import sys
import math
import time
import shutil
import argparse
from collections import deque

import numpy as np
import pygame

import vizdoom as vzd
from vizdoom import DoomGame, Mode, Button, GameVariable

# ─────────────────────────────────────────────────────
# CL SDK setup
# ─────────────────────────────────────────────────────
CL_AVAILABLE = False
try:
    import cl
    from cl.neurons import DummyNeurons  # official dummy neurons
    CL_AVAILABLE = True
except Exception:
    cl = None
    DummyNeurons = None

def cl_open():
    """
    Open a CL neurons session if available, else return a dummy fallback.
    The fallback generates deterministic pseudo-spikes for simulation.
    """
    if CL_AVAILABLE:
        return DummyNeurons()
    else:
        class _FallbackDummy:
            def __enter__(self): return self
            def __exit__(self, et, ev, tb): return False

            def loop(self, ticks_per_second=35):
                dt = 1.0 / max(1, ticks_per_second)
                rng = np.random.default_rng(42)
                while True:
                    class _T:
                        spikes = []
                    tick = _T()
                    tick.analysis = tick
                    yield tick
                    time.sleep(dt)

            def create_stim_plan(self): return self
            def interrupt(self, *_a, **_k): pass
            def stim(self, *_a, **_k): pass
            def run(self): pass

        return _FallbackDummy()

# ─────────────────────────────────────────────────────
# Target configuration
# ─────────────────────────────────────────────────────
ARMOR_X, ARMOR_Y = -32.5, 97.7
ARMOR_RADIUS = 5.0

# Ensure config file exists
bak_file = "custom_room.bak"
cfg_file = "custom_room.cfg"
if not os.path.exists(cfg_file):
    shutil.copyfile(bak_file, cfg_file)

# ─────────────────────────────────────────────────────
# Neuron channel groups
# ─────────────────────────────────────────────────────
FORWARD_GROUP  = {10,11,12,13,14,15,16,17}
BACKWARD_GROUP = {20,21,22,23,24,25}
LEFT_GROUP     = {40,41,42,43,44,45,46,47}
RIGHT_GROUP    = {30,31,32,33,34,35,36,37}

ENC_LEFT  = {9,17,19,29}
ENC_RIGHT = {18,26,38,39}
ENC_CTR   = {33,34}

FB_POS    = {27,28,47,48}
FB_NEG    = {35,36,49,50}

# ─────────────────────────────────────────────────────
# Helper functions for CL neurons
# ─────────────────────────────────────────────────────
def _mk_channels(chs):
    """
    Convert a list of channels into a CL ChannelSet if available.
    """
    if CL_AVAILABLE:
        return cl.ChannelSet(*sorted(chs))
    return tuple(sorted(chs))

def _mk_stim_design():
    """
    Return a default CL stimulation design.
    """
    if not CL_AVAILABLE:
        return None
    try:
        return cl.StimDesign(160, -2.5, 160, +2.5)
    except Exception:
        return cl.StimDesign()

def _mk_burst(count, freq):
    """
    Return a CL burst design (number of pulses and frequency).
    """
    if not CL_AVAILABLE:
        return None
    try:
        return cl.BurstDesign(count, freq)
    except Exception:
        try:
            return cl.BurstDesign(count=count, frequency=freq)
        except Exception:
            return cl.BurstDesign()

def encode_state(neurons, dist, bearing_sign):
    """
    Encode the agent's state to neurons based on distance and direction to target.

    Parameters:
    - neurons: CL neuron session or dummy
    - dist: distance to target
    - bearing_sign: sign of X-axis difference to target
    """
    if not CL_AVAILABLE:
        return

    d = float(min(200.0, max(0.0, dist)))
    f = int(round(5 + (1.0 - d/200.0)*(35-5)))

    target = ENC_CTR
    if bearing_sign < -0.05:
        target = ENC_LEFT
    elif bearing_sign > 0.05:
        target = ENC_RIGHT

    plan = neurons.create_stim_plan()
    plan.interrupt(_mk_channels(target))
    plan.stim(_mk_channels(target), _mk_stim_design(), _mk_burst(120, max(5, f)))
    plan.run()

def feedback_stim(neurons, positive: bool, magnitude: float):
    """
    Provide reinforcement-style feedback to neurons.

    Parameters:
    - positive: True for positive feedback, False for negative
    - magnitude: feedback strength [0,1]
    """
    if not CL_AVAILABLE:
        return

    m = max(0.0, min(1.0, float(magnitude)))
    count = int(40 + 120 * m)
    freq = int(60 + 100 * m)
    chs = FB_POS if positive else FB_NEG

    plan = neurons.create_stim_plan()
    plan.interrupt(_mk_channels(chs))
    plan.stim(_mk_channels(chs), _mk_stim_design(), _mk_burst(count, freq))
    plan.run()

def distance(a_x, a_y, b_x, b_y):
    """
    Euclidean distance between two points.
    """
    return math.hypot(a_x - b_x, a_y - b_y)

def counts_from_spikes(spikes):
    """
    Decode neuron spikes into direction counts.

    Returns a dict with counts for forward/backward/left/right.
    """
    out = dict(forward=0, backward=0, left=0, right=0)
    for s in spikes:
        ch = getattr(s, "channel", None)
        if ch in FORWARD_GROUP:
            out["forward"] += 1
        if ch in BACKWARD_GROUP:
            out["backward"] += 1
        if ch in LEFT_GROUP:
            out["left"] += 1
        if ch in RIGHT_GROUP:
            out["right"] += 1
    return out

# ─────────────────────────────────────────────────────
# Visualization overlay
# ─────────────────────────────────────────────────────
class DoomOverlay:
    """
    Pygame overlay to visualize the agent, goal, neuron activations, and penalties.
    """

    def __init__(self, width=400, height=400, scale=3):
        pygame.init()
        self.width = width
        self.height = height
        self.scale = scale
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Neuron-Controlled Doom Overlay")
        self.font = pygame.font.SysFont("Arial", 14)
        self.clock = pygame.time.Clock()

    def draw(self, px, py, totals, norm_totals, active_dirs, penalties,
             goal=(ARMOR_X, ARMOR_Y), goal_radius=ARMOR_RADIUS):

        self.screen.fill((30, 30, 30))

        # Draw goal
        gx = int(goal[0] * self.scale + self.width // 2)
        gy = int(-goal[1] * self.scale + self.height // 2)
        pygame.draw.circle(self.screen, (0, 255, 0), (gx, gy), int(goal_radius * self.scale), 0)

        # Draw agent
        px_screen = int(px * self.scale + self.width // 2)
        py_screen = int(-py * self.scale + self.height // 2)
        pygame.draw.circle(self.screen, (0, 0, 255), (px_screen, py_screen), 6)

        # Draw directional bars
        directions = ['forward', 'backward', 'left', 'right']
        colors = {'forward': (255, 0, 0), 'backward': (255, 165, 0),
                  'left': (0, 255, 255), 'right': (255, 0, 255)}
        base_x, base_y = 50, 350
        bar_width, spacing = 60, 80

        for i, d in enumerate(directions):
            pygame.draw.rect(self.screen, (50, 50, 50), (base_x + i * spacing, base_y - 100, bar_width, 100))
            max_total = max((v if isinstance(v, (int, float)) else 0 for v in totals.values()), default=0)
            intensity = 0.0 if max_total == 0 else min(1.0, totals.get(d, 0) / max_total)
            pygame.draw.rect(self.screen, colors[d], (base_x + i * spacing, base_y - int(intensity * 100),
                                                      bar_width, int(intensity * 100)))
            pen = penalties.get(d, 0)
            pygame.draw.rect(self.screen, (255, 255, 255), (base_x + i * spacing, base_y - int(intensity * 100),
                                                            bar_width, int(min(pen, 100))))
            prob_text = self.font.render(f"{norm_totals.get(d, 0):.2f}", True, (255, 255, 255))
            self.screen.blit(prob_text, (base_x + i * spacing, base_y + 5))
            if d in active_dirs:
                pygame.draw.rect(self.screen, (0, 255, 0), (base_x + i * spacing, base_y - 100, bar_width, 100), 2)

        pygame.display.flip()
        self.clock.tick(30)

# ─────────────────────────────────────────────────────
# Main control loop
# ─────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Advanced neuron-controlled Doom (custom room)")
    ap.add_argument("--ticks-per-second", type=int, default=35)
    ap.add_argument("--inertia-window", type=int, default=5)
    ap.add_argument("--allow-stim", action="store_true")
    ap.add_argument("--show-logs", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    stim_enabled = bool(args.allow_stim and CL_AVAILABLE)
    rng = np.random.default_rng(args.seed)

    # Initialize Doom
    game = DoomGame()
    game.load_config(cfg_file)
    game.set_window_visible(True)
    game.set_mode(Mode.PLAYER)
    game.set_available_buttons([Button.MOVE_FORWARD, Button.MOVE_BACKWARD, Button.MOVE_LEFT, Button.MOVE_RIGHT])
    game.set_available_game_variables([GameVariable.POSITION_X, GameVariable.POSITION_Y])
    game.init()
    game.new_episode()

    # Initialize inertia, penalties, and overlay
    inertia = {d: deque(maxlen=args.inertia_window) for d in ["forward","backward","left","right"]}
    penalty = {d: 0.0 for d in inertia}
    last_dist = None
    overlay = DoomOverlay(width=400, height=400, scale=3)

    with cl_open() as neurons:
        for tick in neurons.loop(ticks_per_second=args.ticks_per_second):
            if game.is_episode_finished():
                break

            # Get agent position and distance to armor
            px = game.get_game_variable(GameVariable.POSITION_X)
            py = game.get_game_variable(GameVariable.POSITION_Y)
            dist = distance(px, py, ARMOR_X, ARMOR_Y)

            # Encode state to neurons
            if stim_enabled:
                encode_state(neurons, dist, np.sign(ARMOR_X - px))

            # Decode spikes
            if CL_AVAILABLE:
                counts = counts_from_spikes(getattr(getattr(tick,"analysis",None),"spikes",[]))
            else:
                counts = dict(forward=rng.poisson(2),
                              backward=rng.poisson(1),
                              left=rng.poisson(1),
                              right=rng.poisson(1))

            # Update inertia
            for d in inertia:
                inertia[d].append(counts[d])

            # Update penalties and provide feedback
            if last_dist is not None:
                delta = dist - last_dist
                active_dirs = [d for d in inertia if inertia[d][-1] > 0]
                if delta < -0.05:
                    for d in active_dirs:
                        penalty[d] = max(-5.0, penalty[d] - 0.1 * inertia[d][-1])
                    if stim_enabled:
                        feedback_stim(neurons, True, min(1.0, -delta/15.0))
                elif delta > 0.05:
                    for d in active_dirs:
                        penalty[d] = min(10.0, penalty[d] + 0.2 * inertia[d][-1])
                    if stim_enabled:
                        feedback_stim(neurons, False, min(1.0, delta/15.0))
            last_dist = dist

            # Compute weighted totals
            raw_totals = {d: max(0, sum(inertia[d]) - penalty[d]) for d in inertia}
            total_sum = sum(raw_totals.values())
            if total_sum > 0:
                norm_totals = {d: raw_totals[d] / total_sum for d in raw_totals}
            else:
                norm_totals = {d: 1.0 / len(raw_totals) for d in raw_totals}

            # Probabilistic multi-action blending
            dirs = list(norm_totals.keys())
            probs = np.array(list(norm_totals.values()), dtype=float)
            probs /= probs.sum()
            chosen = rng.choice(dirs, p=probs)
            active_dirs = [chosen]
            for d in dirs:
                if d != chosen and norm_totals[d] > 0.3:
                    active_dirs.append(d)

            # Fallback if no spikes
            if not any(norm_totals.values()):
                dx, dy = ARMOR_X - px, ARMOR_Y - py
                active_dirs = []
                if dy > 1: active_dirs.append("forward")
                if dy < -1: active_dirs.append("backward")
                if dx < -1: active_dirs.append("left")
                if dx > 1: active_dirs.append("right")

            # Apply actions with scaled intensity
            action = [1 if "forward" in active_dirs else 0,
                      1 if "backward" in active_dirs else 0,
                      1 if "left" in active_dirs else 0,
                      1 if "right" in active_dirs else 0]
            intensity = max(1, max(raw_totals.values()))
            game.make_action([a * min(1, intensity/5) for a in action], 1)

            # Update overlay
            overlay.draw(px, py, raw_totals, norm_totals, active_dirs, penalty)

            # Logging
            if args.show_logs:
                print(f"pos=({px:.2f},{py:.2f}) d={dist:.2f} counts={counts} totals={raw_totals} "
                      f"norm={norm_totals} act={action} intensity={intensity} pen={penalty}")

            # Check if goal reached
            if dist <= ARMOR_RADIUS:
                if args.show_logs:
                    print("Reached armor!")
                game.make_action([0,0,0,0])
                time.sleep(1)
                break

    game.close()
    print("Done.")

if __name__ == "__main__":
    main()
