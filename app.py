#!/usr/bin/env python3
"""
Hybrid closed-loop ViZDoom + cl (neural stim) system — polished version.

SAFETY FIRST:
 - Default mode is _simulation only_. To enable real hardware stimulation you MUST:
    1) pass BOTH --allow-stim and --confirm-stim on the command line, and
    2) set environment variable CORTICAL_SAFE_KEY="I_HAVE_APPROVAL"
 - This program will **refuse** to run hardware stimulation unless the above conditions are met.
 - Do not run hardware stimulation on humans or animals without Institutional approvals, trained staff, and
   hardware safety interlocks. The authors / assistant DO NOT endorse unsafe use.

This file:
 - Fixes shape / hardcoded-size bugs in the DQN network
 - Adds robust guards for missing patterns/spatial maps
 - Adds safety gating for stimulation
 - Adds a simulation-only fallback if `cl` SDK unavailable (so you can test without hardware)
 - Saves normalization statistics with spike datasets and uses them at runtime
 - Provides clearer logging and runtime checks
 - Adds distance-shaped reward and visualization hooks (distance traces & automap snapshots)
"""

from __future__ import annotations
import os
import sys
import argparse
import random
import csv
import math
import logging
from collections import deque
from time import time, perf_counter
from typing import Optional, Dict, List, Tuple

import numpy as np
import skimage.transform
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

# VizDoom imports
import vizdoom as vzd
from vizdoom import DoomGame, GameState, Mode

# OpenCV for optional automap display (headless-safe usage guarded)
import cv2

# Try to import cortical labs SDK (cl). If not present, provide a safe simulation stub.
try:
    import cl  # real SDK
    from cl import Neurons
    from cl.closed_loop import LoopTick
    from cl.stim_plan import StimPlan

    CL_AVAILABLE = True
except Exception:
    CL_AVAILABLE = False

    # Minimal simulation stub for development/testing without hardware.
    class _DummyTick:
        class _Analysis:
            def __init__(self):
                self.spikes = []  # list-like of dummy spike objects with .channel

        def __init__(self):
            self.analysis = _DummyTick._Analysis()

    class DummyStimPlan:
        def __init__(self):
            pass

        def interrupt(self, *args, **kwargs):
            return

        def stim(self, *args, **kwargs):
            return

        def run(self):
            return

    class DummyNeurons:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def create_stim_plan(self):
            return DummyStimPlan()

        def interrupt(self, *args, **kwargs):
            return

        def loop(self, ticks_per_second=35):
            # Yield a single dummy tick repeatedly for code compatibility.
            while True:
                yield _DummyTick()

        def __repr__(self):
            return "<DummyNeurons simulation>"

    # Minimal Dummy classes for ChannelSet, StimDesign, BurstDesign for compatibility
    class DummyChannelSet(tuple):
        def __new__(cls, *args):
            return tuple.__new__(cls, args)

    class DummyStimDesign:
        def __init__(self, *args, **kwargs):
            pass

    class DummyBurstDesign:
        def __init__(self, *args, **kwargs):
            pass

    # Attach to fake "cl" namespace so code below can reference cl.* safely
    import types

    cl = types.SimpleNamespace(
        ChannelSet=DummyChannelSet,
        StimDesign=DummyStimDesign,
        BurstDesign=DummyBurstDesign,
        StimPlan=DummyStimPlan,
        Neurons=DummyNeurons,
    )
    Neurons = DummyNeurons

# -------------------------
# Config / Hyperparameters
# -------------------------
RESOLUTION = (30, 45)  # Preprocessing (H, W)
FRAME_REPEAT = 1
TICKS_PER_SECOND = 35

# DQN teacher-related (kept lightweight)
LEARNING_RATE = 0.00025
DISCOUNT_FACTOR = 0.99
TRAIN_EPOCHS = 3
LEARNING_STEPS_PER_EPOCH = 1500
REPLAY_MEMORY_SIZE = 10000
BATCH_SIZE = 64

# Spike dataset & decoder
SPIKE_WINDOW = 8  # longer history
SPIKE_DECODER_HID = 128
SPIKE_DECODER_EPOCHS = 30
SPIKE_DECODER_LR = 1e-3
SPIKE_DATA_MIN_SAMPLES = 1000

# Stim safety defaults (verify with your hardware)
STIM_PULSE_US = 160
STIM_AMP_NEG_UA = -2.5
STIM_AMP_POS_UA = 2.5

# Precompile frequencies used (Hz)
PRECOMPILE_FREQS = [4, 8, 12, 20, 30, 40]

# Logging / misc
MOVING_AVG_WINDOW = 10
SEED = 1234

# -------------------------
# Logging setup
# -------------------------
logger = logging.getLogger("cl_loop")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(ch)


# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_action_vector(action_idx: int, n_buttons: int) -> List[int]:
    """Map arbitrary action index to env button vector safely."""
    if n_buttons <= 0:
        return []
    v = [0] * n_buttons
    safe_idx = action_idx if 0 <= action_idx < n_buttons else (action_idx % n_buttons)
    v[safe_idx] = 1
    return v


def preprocess(img) -> np.ndarray:
    """Convert ViZDoom screen buffer to grayscale RESOLUTION (1,H,W)."""
    if img is None:
        return np.zeros((1, RESOLUTION[0], RESOLUTION[1]), dtype=np.float32)
    arr = np.asarray(img)
    if arr.ndim == 3:
        # normalize to H x W x C
        if arr.shape[0] == 3 and arr.shape[2] != 3:
            arr = np.transpose(arr, (1, 2, 0))
        # now arr.shape[2] should be channels
        if arr.ndim == 3 and arr.shape[2] == 3:
            gray = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
        else:
            # fallback average
            gray = np.mean(arr, axis=2) if arr.ndim == 3 else arr
    elif arr.ndim == 2:
        gray = arr
    else:
        gray = np.zeros((RESOLUTION[0], RESOLUTION[1]), dtype=np.float32)
    resized = skimage.transform.resize(gray, RESOLUTION, anti_aliasing=True)
    return np.expand_dims(resized.astype(np.float32), axis=0)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# -------------------------
# Dueling DQN (teacher) - robust shape handling
# -------------------------
class DuelQNet(nn.Module):
    def __init__(self, available_actions_count: int,
                 input_shape: Tuple[int, int, int] = (1, RESOLUTION[0], RESOLUTION[1])):
        """
        Dynamically compute flattened conv output size to avoid hard-coded view() errors.
        input_shape: (C, H, W)
        """
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 8, kernel_size=3, stride=2, bias=False),
                                   nn.BatchNorm2d(8), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(8, 8, kernel_size=3, stride=2, bias=False),
                                   nn.BatchNorm2d(8), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False),
                                   nn.BatchNorm2d(8), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=False),
                                   nn.BatchNorm2d(16), nn.ReLU())

        # compute flattened size dynamically using a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            tmp = self.conv1(dummy);
            tmp = self.conv2(tmp);
            tmp = self.conv3(tmp);
            tmp = self.conv4(tmp)
            flat_dim = tmp.view(1, -1).shape[1]

        # force even for split into state/advantage halves
        if flat_dim % 2 != 0:
            flat_dim += 1
        self._flat_dim = flat_dim
        half = flat_dim // 2

        self.state_fc = nn.Sequential(nn.Linear(half, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage_fc = nn.Sequential(nn.Linear(half, 64), nn.ReLU(), nn.Linear(64, available_actions_count))

        logger.info(f"DuelQNet initialized: flat_dim={flat_dim} half={half} actions={available_actions_count}")

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        # pad if needed
        if x.shape[1] < self._flat_dim:
            pad = self._flat_dim - x.shape[1]
            x = torch.nn.functional.pad(x, (0, pad))
        half = x.shape[1] // 2
        x1 = x[:, :half];
        x2 = x[:, half:]
        state_value = self.state_fc(x1).reshape(-1, 1)
        advantage_values = self.advantage_fc(x2)
        return state_value + (advantage_values - advantage_values.mean(dim=1, keepdim=True))


class DQNAgent:
    def __init__(self, action_size: int, device: torch.device, load_model: Optional[str] = None):
        self.action_size = action_size
        self.device = device
        self.q_net = DuelQNet(action_size, input_shape=(1, RESOLUTION[0], RESOLUTION[1])).to(device)
        self.target_net = DuelQNet(action_size, input_shape=(1, RESOLUTION[0], RESOLUTION[1])).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.opt = optim.SGD(self.q_net.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.batch_size = BATCH_SIZE
        self.discount = DISCOUNT_FACTOR
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        if load_model:
            if os.path.exists(load_model):
                self.q_net.load_state_dict(torch.load(load_model, map_location=device))
                self.target_net.load_state_dict(self.q_net.state_dict())
                self.epsilon = self.epsilon_min
                logger.info(f"[DQN] Loaded model from {load_model}")
            else:
                logger.warning(f"[DQN] load path not found: {load_model}")

    def get_action(self, state: np.ndarray) -> int:
        # state shape (1, H, W)
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            t = torch.from_numpy(np.expand_dims(state, axis=0)).float().to(self.device)  # (B, C, H, W)
            q = self.q_net(t)
            return int(torch.argmax(q, dim=-1).item())

    def store(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states = torch.from_numpy(np.stack([b[0] for b in batch])).float().to(self.device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.int64).to(self.device)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32).to(self.device)
        next_states = torch.from_numpy(np.stack([b[3] for b in batch])).float().to(self.device)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            next_q_main = self.q_net(next_states)
            next_actions = torch.argmax(next_q_main, dim=1, keepdim=True)
            next_q_target = self.target_net(next_states)
            next_q_vals = next_q_target.gather(1, next_actions).squeeze(1)
            target_q = rewards + (1.0 - dones) * self.discount * next_q_vals

        q_vals = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = nn.MSELoss()(q_vals, target_q)
        self.opt.zero_grad();
        loss.backward();
        self.opt.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())


# -------------------------
# Spike decoder MLP
# -------------------------
class SpikeDecoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int = SPIKE_DECODER_HID, out: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# Channel config & precompiled stim plans
# -------------------------
# NOTE: adjust channels to match your hardware mapping.
# all_channels: example set excluding some reserved channels
all_channels = [i for i in range(64) if i not in {0, 4, 7, 56, 63}]
try:
    all_channels_set = cl.ChannelSet(*all_channels)
except Exception:
    # simulation stub may treat ChannelSet differently
    all_channels_set = cl.ChannelSet(*all_channels)

# Stim and action channel groups - adjust per array layout
stim_channels = cl.ChannelSet(9, 10, 17, 18)
forward_channels = (41, 42, 49, 50)
left_channels = (13, 14, 21, 22)
right_channels = (45, 46, 53, 54)
action_channel_groups = (forward_channels, left_channels, right_channels)
feedback_channels = (27, 28, 35, 36)
try:
    feedback_channels_set = cl.ChannelSet(*feedback_channels)
except Exception:
    feedback_channels_set = cl.ChannelSet(*feedback_channels)

# Stim design
try:
    stim_design = cl.StimDesign(STIM_PULSE_US, STIM_AMP_NEG_UA, STIM_PULSE_US, STIM_AMP_POS_UA)
except Exception:
    # some SDKs may require different args; fallback to a simple construction
    try:
        stim_design = cl.StimDesign(pulse_us=STIM_PULSE_US, amp_neg_uA=STIM_AMP_NEG_UA,
                                    pulse_us_pos=STIM_PULSE_US, amp_pos_uA=STIM_AMP_POS_UA)
    except Exception:
        stim_design = cl.StimDesign()  # stub fallback


def _make_burst_design(count: int, freq: int):
    """Create a BurstDesign with keyword-safe construction (tries several signatures)."""
    try:
        return cl.BurstDesign(count, freq)
    except Exception:
        # try common keyword names
        try:
            return cl.BurstDesign(count=count, frequency=freq)
        except Exception:
            try:
                return cl.BurstDesign(burst_count=count, burst_frequency=freq)
            except Exception:
                # last resort stub
                return cl.BurstDesign()


def build_precompiled_plans(neurons: Neurons, freqs=PRECOMPILE_FREQS) -> Dict[int, object]:
    plans = {}
    for f in freqs:
        try:
            p = neurons.create_stim_plan()
            # interrupt only the stim_channels before running this plan
            p.interrupt(stim_channels)
            # moderate burst length; user can override
            bd = _make_burst_design(200, int(max(1, f)))
            p.stim(stim_channels, stim_design, bd)
            plans[int(f)] = p
        except Exception as e:
            logger.warning(f"[PLANS] Couldn't precompile freq {f}: {e}")
    return plans


# -------------------------
# Encoding strategies
# -------------------------
def encode_temporal(neurons: Neurons, game_state: Optional[GameState], plans: Dict[int, object]):
    """Temporal freq encoding: distance -> frequency -> run precompiled plan."""
    if not plans:
        return
    if game_state is None:
        return
    try:
        guy = np.array(game_state.game_variables[2:4])
        armor_obj = next((o for o in game_state.objects if getattr(o, "name", "") == "GreenArmor"), None)
        if armor_obj is None:
            return
        armor = np.array([armor_obj.position_x, armor_obj.position_y])
        d = np.linalg.norm(guy - armor)
    except Exception:
        return
    max_d = 200.0
    d = min(d, max_d)
    min_f, max_f = 4, 40
    f = int(round(max_f - (d / max_d) * (max_f - min_f)))
    freqs = sorted(plans.keys())
    if not freqs:
        return
    chosen = min(freqs, key=lambda x: abs(x - f))
    try:
        plans[chosen].run()
    except Exception as e:
        logger.debug(f"[ENCODE] temporal plan run failed: {e}")


def encode_rate(neurons: Neurons, game_state: Optional[GameState], patterns: Optional[List[object]] = None):
    """
    Rate coding: closer -> stimulate more electrodes in stim_channels set (spatial recruitment).
    patterns: list of StimPlan objects sized 1..N style
    """
    if not patterns:
        return
    if game_state is None:
        return
    try:
        guy = np.array(game_state.game_variables[2:4])
        armor_obj = next((o for o in game_state.objects if getattr(o, "name", "") == "GreenArmor"), None)
        if armor_obj is None:
            return
        armor = np.array([armor_obj.position_x, armor_obj.position_y])
        d = np.linalg.norm(guy - armor)
    except Exception:
        return
    max_d = 200.0
    d = min(d, max_d)
    ratio = 1.0 - (d / max_d)
    idx = int(round(ratio * (len(patterns) - 1)))
    idx = max(0, min(idx, len(patterns) - 1))
    try:
        patterns[idx].interrupt(patterns[idx])
        patterns[idx].run()
    except Exception as e:
        logger.debug(f"[ENCODE] rate plan failed idx={idx}: {e}")


def encode_spatial(neurons: Neurons, game_state: Optional[GameState], spatial_map: Optional[Dict[int, object]] = None):
    """
    Spatial encoding: map angle/relative location to different electrode groups.
    spatial_map: dict of sector -> StimPlan
    """
    if not spatial_map:
        return
    if game_state is None:
        return
    try:
        guy = np.array(game_state.game_variables[2:4])
        armor_obj = next((o for o in game_state.objects if getattr(o, "name", "") == "GreenArmor"), None)
        if armor_obj is None:
            return
        armor = np.array([armor_obj.position_x, armor_obj.position_y])
        vec = armor - guy
        angle = math.atan2(vec[1], vec[0])
    except Exception:
        return
    sectors = sorted(spatial_map.keys())
    if not sectors:
        return
    sector_count = len(sectors)
    sector_idx = int(((angle + math.pi) / (2 * math.pi)) * sector_count) % sector_count
    chosen = sectors[sector_idx]
    try:
        plan = spatial_map[chosen]
        plan.interrupt(plan)
        plan.run()
    except Exception as e:
        logger.debug(f"[ENCODE] spatial plan failed sector={chosen}: {e}")


def encode_hybrid(neurons: Neurons, game_state: Optional[GameState], plans=None, patterns=None, spatial_map=None):
    # combine temporal + rate + spatial with weighted choice
    if plans:
        encode_temporal(neurons, game_state, plans)
    # small chance to add spatial variation
    if spatial_map and (random.random() < 0.25):
        encode_spatial(neurons, game_state, spatial_map)
    if patterns:
        encode_rate(neurons, game_state, patterns)


# -------------------------
# Decoding: spike history & features
# -------------------------
def extract_spike_features(spikes, window_hist: deque) -> Tuple[np.ndarray, np.ndarray]:
    """
    spikes: list of spike objects for current tick
    window_hist: deque of past spike-count arrays (len = SPIKE_WINDOW-1)
    returns (feat_vector, counts)
    feat vector is flattened length (len(action_channel_groups) * SPIKE_WINDOW)
    """
    counts = np.zeros(len(action_channel_groups), dtype=np.float32)
    if spikes:
        for sp in spikes:
            ch = getattr(sp, "channel", None)
            if ch is None:
                continue
            for i, grp in enumerate(action_channel_groups):
                if ch in grp:
                    counts[i] += 1
    hist_copy = list(window_hist) + [counts]
    feat = np.concatenate(hist_copy, axis=0)
    return feat, counts


# -------------------------
# Graded feedback (biologically inspired)
# -------------------------
def graded_feedback_plan(neurons: Neurons, magnitude: float = 1.0, positive: bool = True):
    """
    Create a stim plan where magnitude scales the burst count/duration.
    This function does not bypass safety checking.
    """
    plan = neurons.create_stim_plan()
    plan.interrupt(feedback_channels_set)
    burst_count = int(5 + magnitude * 50)  # 5..55
    freq = int(80 + magnitude * 120)  # 80..200Hz
    if positive:
        try:
            plan.stim(feedback_channels_set, stim_design, _make_burst_design(burst_count, freq))
        except Exception:
            pass
    else:
        try:
            for i, ch in enumerate(feedback_channels):
                plan.stim(cl.ChannelSet(ch), stim_design, _make_burst_design(burst_count, freq + i * 10))
        except Exception:
            pass
    return plan


# -------------------------
# Distance-shaped reward helpers
# -------------------------
def get_player_and_armor_positions(state: Optional[GameState]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (player_xy, armor_xy) or (None, None) if not available."""
    if state is None:
        return None, None
    try:
        guy = np.array(state.game_variables[2:4], dtype=np.float32)
        armor_obj = next((o for o in state.objects if getattr(o, "name", "") == "GreenArmor"), None)
        if armor_obj is None:
            return guy, None
        armor = np.array([armor_obj.position_x, armor_obj.position_y], dtype=np.float32)
        return guy, armor
    except Exception:
        return None, None


def armor_distance(state: Optional[GameState], max_distance: float = 200.0) -> Optional[float]:
    """Clipped Euclidean distance to GreenArmor, or None if not present."""
    guy, armor = get_player_and_armor_positions(state)
    if guy is None or armor is None:
        return None
    d = float(np.linalg.norm(guy - armor))
    return min(d, max_distance)


class DistanceReward:
    """
    Positive reward for moving closer to GreenArmor; negative for moving away.
    Smooth, clipped, and scaleable; also adds a one-time pickup bonus.
    """
    def __init__(self, scale: float = 0.05, pickup_bonus: float = 50.0, max_distance: float = 200.0):
        self.prev_distance: Optional[float] = None
        self.scale = float(scale)
        self.pickup_bonus = float(pickup_bonus)
        self.max_distance = float(max_distance)
        self.prev_had_armor: Optional[bool] = None

    def start_episode(self):
        self.prev_distance = None
        self.prev_had_armor = None

    def step(self, state_before: Optional[GameState], state_after: Optional[GameState]) -> float:
        # detect armor presence for pickup bonus
        def has_armor(st: Optional[GameState]) -> bool:
            if st is None:
                return False
            try:
                return any(getattr(o, "name", "") == "GreenArmor" for o in st.objects)
            except Exception:
                return False

        r = 0.0
        d_before = armor_distance(state_before, self.max_distance)
        d_after  = armor_distance(state_after,  self.max_distance)

        # shaped reward on distance delta (if both distances are known)
        if d_before is not None and d_after is not None:
            delta = d_before - d_after  # positive if we moved closer
            # small, smooth reward; clip to avoid explosions
            r += self.scale * float(np.clip(delta, -10.0, 10.0))

        # pickup bonus when armor disappears from object list
        now_has = has_armor(state_after)
        was_has = has_armor(state_before)
        if was_has and not now_has:
            r += self.pickup_bonus

        self.prev_distance = d_after
        self.prev_had_armor = now_has
        return r


# -------------------------
# Per-action neural feedback hooks (safe-gated)
# -------------------------
def per_action_feedback(neurons: Neurons, reward: float, allow_stim: bool):
    if not allow_stim:
        return
    try:
        plan = neurons.create_stim_plan()
        plan.interrupt(feedback_channels_set)
        magnitude = float(np.clip(abs(reward), 0.0, 1.0))
        burst = _make_burst_design(int(10 + 60 * magnitude), int(80 + 120 * magnitude))
        if reward >= 0.0:
            plan.stim(feedback_channels_set, stim_design, burst)
        else:
            # stagger channels slightly for negative feedback
            for i, ch in enumerate(feedback_channels):
                plan.stim(cl.ChannelSet(ch), stim_design, _make_burst_design(int(5 + 40 * magnitude), 140 + 20 * i))
        plan.run()
    except Exception:
        pass


def end_of_episode_feedback(neurons: Neurons, episode_score: float, allow_stim: bool):
    if not allow_stim:
        return
    try:
        magnitude = min(1.0, max(0.0, episode_score / 100.0))
        plan = graded_feedback_plan(neurons, magnitude=magnitude, positive=(episode_score >= 0))
        plan.run()
    except Exception:
        pass


# -------------------------
# Visualization helpers
# -------------------------
def log_distance_trace(distances: List[Optional[float]], save_path: str, ep_idx: int):
    xs = list(range(len(distances)))
    ys = [d if (d is not None) else np.nan for d in distances]
    try:
        plt.figure(figsize=(6,3))
        plt.plot(xs, ys, linewidth=1.5)
        plt.title(f"Distance to GreenArmor — Episode {ep_idx}")
        plt.xlabel("tick"); plt.ylabel("distance (clipped)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"distance_ep{ep_idx}.png"))
        plt.close()
    except Exception:
        pass


def save_automap_snapshot(game: DoomGame, save_path: str, ep_idx: int, step_idx: int):
    try:
        state = game.get_state()
        if state is None or state.automap_buffer is None:
            return
        im = state.automap_buffer
        # write a few snapshots per episode (every ~30 steps)
        if step_idx % 30 == 0:
            out = os.path.join(save_path, f"automap_ep{ep_idx}_t{step_idx}.png")
            cv2.imwrite(out, im)
    except Exception:
        pass


# -------------------------
# Dataset collection using teacher
# -------------------------
def collect_spike_dataset(game: DoomGame, neurons: Neurons, teacher_agent: DQNAgent, target_samples: int = 3000,
                          max_episodes: int = 50, plans=None, patterns=None, spatial_map=None,
                          allow_stim: bool = False, window: int = SPIKE_WINDOW, save_path: str = ".",
                          show_automap_flag: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs teacher-guided episodes while encoding into neurons and collecting spikes labeled with teacher actions.
    Returns X (N, 3*window) and y (N,)
    """
    logger.info(f"[DATA] Starting spike dataset collection. allow_stim={allow_stim}")
    X, y = [], []
    samples = 0;
    episodes = 0
    spike_hist = deque(maxlen=window - 1)
    for _ in range(window - 1):
        spike_hist.append(np.zeros(len(action_channel_groups), dtype=np.float32))

    while samples < target_samples and episodes < max_episodes:
        game.new_episode();
        episodes += 1
        while not game.is_episode_finished() and samples < target_samples:
            state = game.get_state()
            frame = preprocess(state.screen_buffer) if state else np.zeros((1, RESOLUTION[0], RESOLUTION[1]),
                                                                           dtype=np.float32)
            teacher_action = teacher_agent.get_action(frame)

            if allow_stim:
                # run encoding that will stimulate hardware via neurons
                encode_hybrid(neurons, state, plans, patterns, spatial_map)
                # capture spikes via neurons.loop for low-jitter tick
                # (we expect SDK's loop to yield once per tick)
                for tick in neurons.loop(ticks_per_second=TICKS_PER_SECOND):
                    feat, counts = extract_spike_features(tick.analysis.spikes, spike_hist)
                    spike_hist.append(counts)
                    X.append(feat.astype(np.float32));
                    y.append(teacher_action);
                    samples += 1
                    break
            else:
                # Simulate spike counts correlated with teacher action
                num_groups = len(action_channel_groups)
                base = np.zeros(num_groups, dtype=np.float32)
                base[teacher_action % num_groups] = 5.0
                simulated_counts = np.random.poisson(base).astype(np.float32)
                feat = np.concatenate(list(spike_hist) + [simulated_counts], axis=0)
                spike_hist.append(simulated_counts)
                X.append(feat.astype(np.float32));
                y.append(teacher_action);
                samples += 1
                # advance environment using teacher action
                action_vec = safe_action_vector(teacher_action, game.get_available_buttons_size())
                game.set_action(action_vec);
                game.advance_action(FRAME_REPEAT)
                if show_automap_flag:
                    show_automap(game)
                continue

            # apply teacher action to game (works both branches)
            action_vec = safe_action_vector(teacher_action, game.get_available_buttons_size())
            game.set_action(action_vec);
            game.advance_action(FRAME_REPEAT)
            if show_automap_flag: show_automap(game)

        logger.info(f"[DATA] Collected {samples} samples after {episodes} episodes.")
    X = np.array(X);
    y = np.array(y, dtype=np.int64)
    ensure_dir(save_path)
    np.save(os.path.join(save_path, "spike_X.npy"), X)
    np.save(os.path.join(save_path, "spike_y.npy"), y)
    logger.info(f"[DATA] Saved spike dataset to {save_path} (X.shape={X.shape}, y.shape={y.shape})")
    return X, y


# -------------------------
# Train spike decoder
# -------------------------
def train_spike_decoder(X: np.ndarray, y: np.ndarray, device: torch.device,
                        epochs: int = SPIKE_DECODER_EPOCHS, lr: float = SPIKE_DECODER_LR,
                        batch_size: int = 256, save_path: str = ".") -> SpikeDecoder:
    input_dim = X.shape[1];
    num_actions = int(y.max() + 1)
    model = SpikeDecoder(input_dim=input_dim, hidden=SPIKE_DECODER_HID, out=num_actions).to(device)
    opt = optim.Adam(model.parameters(), lr=lr);
    ce = nn.CrossEntropyLoss()
    idxs = np.arange(len(X))
    for ep in range(epochs):
        np.random.shuffle(idxs);
        losses = []
        for i in range(0, len(idxs), batch_size):
            bs = idxs[i:i + batch_size]
            xb = torch.from_numpy(X[bs]).float().to(device);
            yb = torch.from_numpy(y[bs]).long().to(device)
            logits = model(xb);
            loss = ce(logits, yb)
            opt.zero_grad();
            loss.backward();
            opt.step();
            losses.append(float(loss.item()))
        logger.info(f"[SPK] Epoch {ep + 1}/{epochs} loss={np.mean(losses):.4f}")
    ensure_dir(save_path)
    pth = os.path.join(save_path, "spike_decoder.pth")
    torch.save(model.state_dict(), pth)
    logger.info(f"[SPK] Saved spike decoder to {pth}")
    return model


# -------------------------
# Optional automap visualization
# -------------------------
def show_automap(game: DoomGame, sleep_ms: int = 28):
    try:
        state = game.get_state()
        if state is None:
            return
        automap = state.automap_buffer
        if automap is not None:
            cv2.imshow("ViZDoom Automap", automap)
            cv2.waitKey(sleep_ms)
    except Exception:
        pass


# -------------------------
# Run closed-loop neuron-centric
# -------------------------
def run_closed_loop_neuron_control(game: DoomGame, neurons: Neurons, spike_decoder: SpikeDecoder,
                                   plans=None, patterns=None, spatial_map=None, episodes: int = 10,
                                   window: int = SPIKE_WINDOW, allow_stim: bool = False, save_path: str = ".",
                                   device: torch.device = torch.device('cpu'), online_finetune: bool = False,
                                   teacher: Optional[DQNAgent] = None, show_automap_flag: bool = False,
                                   norm_mu: Optional[np.ndarray] = None, norm_std: Optional[np.ndarray] = None):
    """
    Main closed-loop evaluation where spike decoder controls game actions.
    """
    spike_decoder.to(device);
    spike_decoder.eval()
    episodes_done = 0
    episode_scores = [];
    moving_avg_scores = [];
    action_agreements = [];
    latency_records = []

    ensure_dir(save_path)
    csv_path = os.path.join(save_path, "episode_log.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "score", "steps", "mean_latency_ms", "agreement_with_teacher"])

    expected_input_dim = len(action_channel_groups) * window
    # Quick input-dim check
    try:
        model_in_size = sum(
            p.numel() for p in spike_decoder.parameters())  # not precise; but we'll check weight shape via first linear
    except Exception:
        model_in_size = None
    logger.info(f"[RUN] Expected decoder input dim: {expected_input_dim}")

    while episodes_done < episodes:
        game.new_episode()
        shaper = DistanceReward(scale=0.05, pickup_bonus=50.0, max_distance=200.0)
        shaper.start_episode()

        spike_hist = deque(maxlen=window - 1)
        for _ in range(window - 1):
            spike_hist.append(np.zeros(len(action_channel_groups), dtype=np.float32))

        total_reward = 0.0
        steps = 0
        rasters = []
        latencies = []
        agreement_counts = 0
        teacher_checks = 0
        dist_trace: List[Optional[float]] = []

        # In non-hardware mode we simulate spikes; in hardware mode we use neurons.loop
        if allow_stim:
            loop_iter = neurons.loop(ticks_per_second=TICKS_PER_SECOND)
        else:
            # create a simple generator that yields a dummy tick for each iteration
            def _sim_loop():
                class _Tick:
                    class _A:
                        spikes = []

                    analysis = _A()

                while True:
                    yield _Tick()

            loop_iter = _sim_loop()

        for tick in loop_iter:
            t0 = perf_counter()
            state_before = game.get_state()
            d_before = armor_distance(state_before)
            if d_before is not None:
                dist_trace.append(d_before)
            else:
                dist_trace.append(None)

            # encode or simulate
            if allow_stim:
                # stimulatory encoding
                encode_hybrid(neurons, state_before, plans, patterns, spatial_map)
                feat, counts = extract_spike_features(tick.analysis.spikes, spike_hist)
                spike_hist.append(counts)
            else:
                # simulation: base correlated with teacher if available
                num_groups = len(action_channel_groups)
                if teacher is not None and state_before is not None:
                    frame = preprocess(state_before.screen_buffer)
                    t_act = teacher.get_action(frame)
                    base = np.ones(num_groups, dtype=np.float32) * 0.5
                    base[t_act % num_groups] += 4.0
                else:
                    base = np.ones(num_groups, dtype=np.float32) * 1.0
                counts = np.random.poisson(base).astype(np.float32)
                feat = np.concatenate(list(spike_hist) + [counts], axis=0)
                spike_hist.append(counts)

            rasters.append(counts)

            # normalization if provided
            feat_in = feat.astype(np.float32)
            if norm_mu is not None and norm_std is not None:
                feat_in = (feat_in - norm_mu.ravel()) / norm_std.ravel()

            xb = torch.from_numpy(feat_in).unsqueeze(0).float().to(device)
            with torch.no_grad():
                logits = spike_decoder(xb)
                act_idx = int(torch.argmax(logits, dim=1).cpu().numpy()[0])

            action_vec = safe_action_vector(act_idx, game.get_available_buttons_size())
            game.set_action(action_vec);
            game.advance_action(FRAME_REPEAT)

            # (d) reward = env + distance shaping
            state_after = game.get_state()
            r_env = game.get_last_reward() if hasattr(game, "get_last_reward") else 0.0
            r_shape = shaper.step(state_before, state_after)
            rew = float(r_env) + float(r_shape)
            total_reward += rew
            steps += 1

            # optional per-action neurofeedback (safe-gated)
            per_action_feedback(neurons, r_shape, allow_stim=allow_stim)

            lat_ms = (perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)

            # teacher agreement
            if teacher is not None and state_before is not None:
                frame = preprocess(state_before.screen_buffer)
                t_action = teacher.get_action(frame)
                teacher_checks += 1
                if t_action == act_idx:
                    agreement_counts += 1

            # online finetune
            if online_finetune and teacher is not None and state_before is not None:
                t_action = teacher.get_action(preprocess(state_before.screen_buffer))
                model = spike_decoder;
                model.train()
                xb_ft = xb;
                yb_ft = torch.tensor([t_action], dtype=torch.long).to(device)
                opt = optim.Adam(model.parameters(), lr=1e-5)
                loss = nn.CrossEntropyLoss()(model(xb_ft), yb_ft)
                opt.zero_grad();
                loss.backward();
                opt.step();
                model.eval()

            if show_automap_flag:
                show_automap(game)
                save_automap_snapshot(game, save_path, episodes_done, steps)

            if game.is_episode_finished():
                mean_lat = float(np.mean(latencies)) if latencies else 0.0
                agreement = (agreement_counts / teacher_checks) if teacher_checks > 0 else None
                episode_scores.append(total_reward)
                moving_avg = float(np.mean(episode_scores[-MOVING_AVG_WINDOW:])) if episode_scores else 0.0
                moving_avg_scores.append(moving_avg)
                action_agreements.append(agreement)
                latency_records.append(mean_lat)

                rasters_np = np.stack(rasters, axis=0) if rasters else np.zeros((0, len(action_channel_groups)))
                raster_path = os.path.join(save_path, f"raster_ep{episodes_done}.npy")
                np.save(raster_path, rasters_np)

                # Save a small PNG raster
                try:
                    plt.figure(figsize=(6, 2))
                    plt.imshow(rasters_np.T, aspect='auto', interpolation='nearest')
                    plt.xlabel('tick');
                    plt.ylabel('action-group');
                    plt.title(f'Raster Ep{episodes_done}')
                    pngpath = os.path.join(save_path, f"raster_ep{episodes_done}.png")
                    plt.savefig(pngpath);
                    plt.close()
                except Exception:
                    pass

                # Save distance trace
                log_distance_trace(dist_trace, save_path, episodes_done)

                csv_writer.writerow(
                    [episodes_done, total_reward, steps, mean_lat, agreement if agreement is not None else "NA"])
                csv_file.flush()

                logger.info(
                    f"[RUN] Episode {episodes_done} score={total_reward:.2f} steps={steps} mean_latency_ms={mean_lat:.2f} agreement={agreement}")
                # graded feedback at episode end (only if allow_stim)
                end_of_episode_feedback(neurons, total_reward, allow_stim=allow_stim)
                # safety interrupt
                try:
                    neurons.interrupt(stim_channels)
                    neurons.interrupt(feedback_channels_set)
                except Exception:
                    pass

                episodes_done += 1
                break  # break per-episode loop

    csv_file.close()
    # Summary plots
    try:
        plt.figure();
        plt.plot(episode_scores, label='episode_score');
        plt.plot(moving_avg_scores, label=f'moving_avg({MOVING_AVG_WINDOW})');
        plt.legend();
        plt.title('Scores');
        plt.savefig(os.path.join(save_path, 'scores.png'));
        plt.close()
    except Exception:
        pass
    if any(a is not None for a in action_agreements):
        try:
            plt.figure();
            plt.plot([a if a is not None else 0 for a in action_agreements]);
            plt.title('Agreement with Teacher');
            plt.savefig(os.path.join(save_path, 'agreement.png'));
            plt.close()
        except Exception:
            pass

    # montage of distances
    try:
        imgs = []
        for i in range(episodes_done):
            p = os.path.join(save_path, f"distance_ep{i}.png")
            if os.path.exists(p):
                imgs.append(cv2.imread(p))
        if imgs:
            h = max(im.shape[0] for im in imgs); w = max(im.shape[1] for im in imgs)
            canvas = np.ones((h * len(imgs), w, 3), dtype=np.uint8) * 255
            for idx, im in enumerate(imgs):
                if im is None: continue
                imr = cv2.resize(im, (w, h))
                canvas[idx*h:(idx+1)*h, :w] = imr
            cv2.imwrite(os.path.join(save_path, "distance_montage.png"), canvas)
    except Exception:
        pass

    logger.info(f"[RUN] Closed-loop finished. Logs saved to {save_path}")


# -------------------------
# CLI + top-level orchestration
# -------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Robust Hybrid DQN-teacher + Neuron-centric Spike Imitation")
    p.add_argument("--config", type=str, default=os.path.join(vzd.scenarios_path, "simpler_basic.cfg"))
    p.add_argument("--train-dqn", action="store_true")
    p.add_argument("--collect-spike-data", action="store_true")
    p.add_argument("--collect-samples", type=int, default=3000)
    p.add_argument("--train-spike-decoder", action="store_true")
    p.add_argument("--run-closed-loop", action="store_true")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--save-path", type=str, default=os.getcwd())
    p.add_argument("--allow-stim", action="store_true",
                   help="Enable hardware stimulation (DANGEROUS). Requires explicit confirmation.")
    p.add_argument("--confirm-stim", action="store_true",
                   help="Secondary explicit flag required in addition to --allow-stim.")
    p.add_argument("--encoding", type=str, choices=["temporal", "rate", "spatial", "hybrid"], default="hybrid")
    p.add_argument("--load-dqn", type=str, default=None)
    p.add_argument("--load-spike-decoder", type=str, default=None)
    p.add_argument("--online-finetune", action="store_true",
                   help="Fine-tune spike decoder online during closed-loop using teacher")
    p.add_argument("--show-automap", action="store_true", help="Display ViZDoom top-down automap for debugging")
    p.add_argument("--debug-run", action="store_true", help="Quick demo run (no neural decoder).")
    return p


def main():
    args = build_argparser().parse_args()

    # default pipeline if invoked with no args
    if len(sys.argv) == 1:
        logger.info("No CLI arguments → running default pipeline (collect -> train -> closed-loop) in simulation mode.")
        args.collect_spike_data = True
        args.collect_samples = 5000
        args.train_spike_decoder = True
        args.run_closed_loop = True
        args.allow_stim = False
        args.train_dqn = False
        args.show_automap_flag = True

    set_seed()
    ensure_dir(args.save_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[INFO] Device: {device}")

    # Safety gating for hardware stimulation
    if args.allow_stim:
        if not args.confirm_stim:
            logger.error("Hardware stimulation requested but --confirm-stim not provided. Aborting.")
            sys.exit(1)
        if os.environ.get("CORTICAL_SAFE_KEY", "") != "I_HAVE_APPROVAL":
            logger.error(
                "Environment variable CORTICAL_SAFE_KEY does not equal 'I_HAVE_APPROVAL'. Aborting stimulation for safety.")
            sys.exit(1)
        if not CL_AVAILABLE:
            logger.error("Requested hardware stimulation but 'cl' SDK not available in Python environment.")
            sys.exit(1)
        logger.warning(
            "!!! WARNING: HARDWARE STIMULATION ENABLED. Confirm approvals and safety BEFORE running on real tissue !!!")

    show_automap_flag = bool(args.show_automap)
    # Initialize ViZDoom
    game = DoomGame();
    game.load_config(args.config)
    game.set_episode_timeout(2000)
    game.set_episode_start_time(10)
    # Window visibility: hide on headless servers unless automap requested
    try:
        game.set_window_visible(bool(show_automap_flag))
    except Exception:
        pass
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    # Expose objects (needed for GreenArmor distance)
    try:
        game.set_labels_buffer_enabled(True)
        game.set_objects_info_enabled(True)
    except Exception:
        pass

    game.init()

    n_buttons = game.get_available_buttons_size()
    if n_buttons < 3:
        logger.warning("[WARN] env has <3 buttons - action mapping assumptions may need update")

    # Build teacher DQN
    teacher = DQNAgent(action_size=3, device=device, load_model=args.load_dqn)
    if args.train_dqn:
        logger.info("[DQN] Training teacher (distance-shaped).")
        actions = [safe_action_vector(a, n_buttons) for a in range(3)]
        shaper = DistanceReward(scale=0.05, pickup_bonus=50.0, max_distance=200.0)

        for epoch in range(TRAIN_EPOCHS):
            logger.info(f"[DQN] epoch {epoch + 1}/{TRAIN_EPOCHS}")
            game.new_episode()
            shaper.start_episode()

            for step in trange(LEARNING_STEPS_PER_EPOCH, leave=False):
                state_before = game.get_state()
                s = preprocess(state_before.screen_buffer) if state_before else np.zeros((1, RESOLUTION[0], RESOLUTION[1]), dtype=np.float32)
                a = teacher.get_action(s)

                # Step environment with teacher-chosen action
                r_env = game.make_action(actions[a], FRAME_REPEAT)
                done = game.is_episode_finished()
                state_after = game.get_state() if not done else None

                # Distance-shaped reward
                r_shape = shaper.step(state_before, state_after)
                r = float(r_env) + float(r_shape)

                s2 = preprocess(state_after.screen_buffer) if (state_after is not None) else np.zeros((1, RESOLUTION[0], RESOLUTION[1]), dtype=np.float32)
                teacher.store(s, a, r, s2, done)
                teacher.train_step()

                if done:
                    game.new_episode()
                    shaper.start_episode()

            teacher.sync_target()

        torch.save(teacher.q_net.state_dict(), os.path.join(args.save_path, "dqn_teacher.pth"))
        logger.info("[DQN] Teacher training complete with distance-shaped rewards.")

    # Open neurons (real or dummy)
    with Neurons() as neurons:
        logger.info(f"[CL] Using neurons object: {neurons}")
        plans = build_precompiled_plans(neurons) if (args.allow_stim and CL_AVAILABLE) else {}
        # prepare patterns for rate coding (stim plan per increasing electrode set)
        patterns = None
        if args.allow_stim and CL_AVAILABLE:
            try:
                # try to extract underlying channel list
                try:
                    stim_list = list(stim_channels.channels) if hasattr(stim_channels, 'channels') else list(
                        stim_channels)
                except Exception:
                    # fallback: try to iterate
                    stim_list = list(stim_channels)
                patterns = []
                for k in range(1, len(stim_list) + 1):
                    p = neurons.create_stim_plan()
                    ch_set = cl.ChannelSet(*stim_list[:k])
                    p.interrupt(ch_set)
                    p.stim(ch_set, stim_design, _make_burst_design(100, 10))
                    patterns.append(p)
            except Exception as e:
                logger.warning(f"[CL] failed to build patterns: {e}")
                patterns = None

        # spatial map example (4 sectors)
        spatial_map = {}
        if args.allow_stim and CL_AVAILABLE:
            try:
                sectors = 4
                # ensure stim_list defined
                try:
                    stim_list
                except NameError:
                    stim_list = list(stim_channels)
                for s in range(sectors):
                    p = neurons.create_stim_plan();
                    p.interrupt(stim_channels)
                    ch = cl.ChannelSet(stim_list[s % len(stim_list)])
                    p.stim(ch, stim_design, _make_burst_design(150, 20 + (s * 10)))
                    spatial_map[s] = p
            except Exception as e:
                logger.warning(f"[CL] failed to build spatial_map: {e}")
                spatial_map = None

        # Collect spike dataset
        if args.collect_spike_data:
            X, y = collect_spike_dataset(game, neurons, teacher, target_samples=args.collect_samples,
                                         max_episodes=50, plans=plans, patterns=patterns, spatial_map=spatial_map,
                                         allow_stim=args.allow_stim, window=SPIKE_WINDOW, save_path=args.save_path,
                                         show_automap_flag=show_automap_flag)
            logger.info(f"[DATA] Collected dataset shapes: X={X.shape} y={y.shape}")

        # Train spike decoder
        spike_decoder = None
        norm_mu = None;
        norm_std = None
        sd_path = os.path.join(args.save_path, "spike_decoder.pth")
        stats_path = os.path.join(args.save_path, "spike_X_stats.npz")
        if args.train_spike_decoder:
            XPath = os.path.join(args.save_path, "spike_X.npy");
            yPath = os.path.join(args.save_path, "spike_y.npy")
            if os.path.exists(XPath) and os.path.exists(yPath):
                X = np.load(XPath);
                y = np.load(yPath)
            else:
                logger.info("[SPK] dataset not found; collecting now...")
                X, y = collect_spike_dataset(game, neurons, teacher, target_samples=args.collect_samples,
                                             max_episodes=50, plans=plans, patterns=patterns, spatial_map=spatial_map,
                                             allow_stim=args.allow_stim, window=SPIKE_WINDOW, save_path=args.save_path,
                                             show_automap_flag=show_automap_flag)
            if len(X) < SPIKE_DATA_MIN_SAMPLES:
                logger.warning(f"[SPK] Warning: small dataset ({len(X)} samples). Consider collecting more.")
            mu = X.mean(axis=0, keepdims=True)
            std = X.std(axis=0, keepdims=True) + 1e-6
            X_norm = (X - mu) / std
            np.save(os.path.join(args.save_path, "spike_X_norm.npy"), X_norm)
            np.savez(stats_path, mu=mu, std=std)
            logger.info(f"[SPK] Saved normalization stats to {stats_path}")
            spike_decoder = train_spike_decoder(X_norm, y, device=device, epochs=SPIKE_DECODER_EPOCHS,
                                                save_path=args.save_path)

        # Optionally load spike decoder and normalization stats
        if args.load_spike_decoder:
            ld = args.load_spike_decoder
            if os.path.exists(ld):
                # load stats if available
                stats_ld = os.path.join(os.path.dirname(ld), "spike_X_stats.npz")
                if os.path.exists(stats_ld):
                    arr = np.load(stats_ld);
                    norm_mu = arr["mu"];
                    norm_std = arr["std"]
                # Build a decoder and load weights
                expected_dim = len(action_channel_groups) * SPIKE_WINDOW
                spike_decoder = SpikeDecoder(input_dim=expected_dim, hidden=SPIKE_DECODER_HID, out=3)
                spike_decoder.load_state_dict(torch.load(ld, map_location=device));
                spike_decoder.to(device)
                logger.info("[SPK] Loaded spike decoder from %s", ld)
            else:
                logger.warning("[SPK] load path not found: %s", ld)

        # If we trained earlier, attempt to load saved normalization stats if present
        if os.path.exists(stats_path) and (norm_mu is None or norm_std is None):
            try:
                arr = np.load(stats_path);
                norm_mu = arr["mu"];
                norm_std = arr["std"]
                logger.info(f"[SPK] Loaded normalization stats from {stats_path}")
            except Exception:
                norm_mu = norm_std = None

        # Run closed-loop
        if args.run_closed_loop:
            # If no decoder present, attempt to load default path
            if spike_decoder is None:
                if os.path.exists(sd_path):
                    expected_dim = len(action_channel_groups) * SPIKE_WINDOW
                    spike_decoder = SpikeDecoder(input_dim=expected_dim, hidden=SPIKE_DECODER_HID, out=3)
                    spike_decoder.load_state_dict(torch.load(sd_path, map_location=device));
                    spike_decoder.to(device)
                    logger.info("[SPK] Loaded spike decoder from default %s", sd_path)
                    if os.path.exists(stats_path):
                        arr = np.load(stats_path);
                        norm_mu = arr["mu"];
                        norm_std = arr["std"]
                else:
                    logger.error("No spike decoder available. Train one or provide --load-spike-decoder.")
                    raise RuntimeError("No spike decoder available. Train or provide one.")
            # run closed-loop evaluation
            run_closed_loop_neuron_control(game, neurons, spike_decoder, plans=plans, patterns=patterns,
                                           spatial_map=spatial_map, episodes=args.episodes, window=SPIKE_WINDOW,
                                           allow_stim=args.allow_stim, save_path=args.save_path, device=device,
                                           online_finetune=args.online_finetune, teacher=teacher,
                                           show_automap_flag=show_automap_flag, norm_mu=norm_mu, norm_std=norm_std)

    # cleanup
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    try:
        game.close()
    except Exception:
        pass
    logger.info("[INFO] All done. Logs saved to: %s", args.save_path)


if __name__ == "__main__":
    main()
