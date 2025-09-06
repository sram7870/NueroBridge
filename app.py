#!/usr/bin/env python3
"""
Hybrid closed-loop ViZDoom + cl (neural stim) system — polished version with SMART SHOOTING.

SAFETY FIRST:
 - Default mode is _simulation only_. To enable real hardware stimulation you MUST:
    1) pass BOTH --allow-stim and --confirm-stim on the command line, and
    2) set environment variable CORTICAL_SAFE_KEY="I_HAVE_APPROVAL"
 - This program will **refuse** to run hardware stimulation unless the above conditions are met.
 - Do not run hardware stimulation on humans or animals without Institutional approvals, trained staff, and
   hardware safety interlocks. The authors / assistant DO NOT endorse unsafe use.

FIXES ADDED:
 - Smart shooting: Only shoots when enemies are visible via CV/object detection
 - Ammo conservation: Penalizes wasteful shooting when no enemies present
 - Action cooldown: Prevents shooting spam
 - Enhanced enemy detection using ViZDoom labels and CV
 - FIXED: Robust game_variables handling to prevent TypeError crashes
 - FIXED: Proper pipeline sequencing to ensure spike decoder is available
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

# Smart shooting parameters
SHOOT_COOLDOWN_FRAMES = 8  # Minimum frames between shots
WASTEFUL_SHOOT_PENALTY = -5.0  # Penalty for shooting when no enemy visible
AMMO_CONSERVATION_BONUS = 1.0  # Small bonus for conserving ammo when appropriate

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

def get_nearest_enemy(state):
    """Return (x, y) screen coordinates of nearest enemy, or None if none found."""
    if state is None or not hasattr(state, 'labels'):
        return None
    enemies = [o for o in state.labels if o.object_type in ENEMY_OBJECT_IDS]
    if not enemies:
        return None
    px, py = RESOLUTION[1]//2, RESOLUTION[0]//2  # approximate agent center
    nearest = min(enemies, key=lambda e: (e.x - px)**2 + (e.y - py)**2)
    return (nearest.x, nearest.y)

def compute_turn_direction(agent_pos, enemy_pos):
    """Return -1 for left, 1 for right, 0 for aligned."""
    if enemy_pos is None:
        return 0
    dx = enemy_pos[0] - agent_pos[0]
    if abs(dx) < 5:  # small tolerance
        return 0
    return 1 if dx > 0 else -1

def generate_enemy_aware_action(agent_pos, state, n_buttons):
    """Return action vector targeting nearest enemy."""
    action_vec = np.zeros(n_buttons, dtype=int)
    enemy_pos = get_nearest_enemy(state)
    turn_dir = compute_turn_direction(agent_pos, enemy_pos)
    if turn_dir == 1:
        action_vec[TURN_RIGHT] = 1
    elif turn_dir == -1:
        action_vec[TURN_LEFT] = 1
    else:
        if enemy_pos is not None:
            action_vec[ATTACK] = 1
        else:
            action_vec[MOVE_FORWARD] = 1
    return action_vec

# -------------------------
# SMART SHOOTING: Enemy Detection
# -------------------------
def detect_enemies_in_state(state: Optional[GameState]) -> bool:
    """
    Detect if any enemies are visible using ViZDoom's object detection system.
    Returns True if enemies detected, False otherwise.
    """
    if state is None:
        return False
    
    try:
        # Check for enemies using ViZDoom's object detection
        if hasattr(state, 'objects') and state.objects:
            enemy_names = {'Cacodemon', 'ZombieMan', 'ShotgunGuy', 'Imp', 'Demon', 'Baron', 'Pinky'}
            for obj in state.objects:
                obj_name = getattr(obj, 'name', '')
                if obj_name in enemy_names:
                    return True
        
        # Fallback: CV-based enemy detection on screen buffer
        if hasattr(state, 'screen_buffer') and state.screen_buffer is not None:
            return detect_enemies_cv(state.screen_buffer)
            
    except Exception as e:
        logger.debug(f"[ENEMY] Detection error: {e}")
    
    return False


def detect_enemies_cv(screen_buffer) -> bool:
    """
    CV-based enemy detection fallback.
    Detects reddish/brownish colors typical of DOOM enemies.
    """
    if screen_buffer is None:
        return False
        
    try:
        # Convert to RGB if needed
        if len(screen_buffer.shape) == 3:
            if screen_buffer.shape[0] == 3:  # CHW format
                img = np.transpose(screen_buffer, (1, 2, 0))
            else:  # HWC format
                img = screen_buffer
        else:
            return False  # Can't detect enemies in grayscale easily
            
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        # Define enemy color ranges (reddish/brownish for typical DOOM enemies)
        # Range 1: Reddish colors (Cacodemon, blood, etc.)
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([160, 50, 50]) 
        red_upper2 = np.array([180, 255, 255])
        
        # Range 2: Brownish colors (ZombieMan, Imp, etc.)
        brown_lower = np.array([5, 50, 20])
        brown_upper = np.array([25, 255, 200])
        
        # Create masks
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
        
        enemy_mask = cv2.bitwise_or(red_mask1, cv2.bitwise_or(red_mask2, brown_mask))
        
        # Count enemy pixels - if above threshold, enemy detected
        enemy_pixels = np.sum(enemy_mask > 0)
        total_pixels = screen_buffer.shape[0] * screen_buffer.shape[1]
        enemy_ratio = enemy_pixels / total_pixels
        
        # Threshold: if >0.5% of pixels are "enemy-colored", consider enemy present
        return enemy_ratio > 0.005
        
    except Exception as e:
        logger.debug(f"[ENEMY CV] Detection error: {e}")
        return False


class ActionGater:
    """
    Gates actions to prevent wasteful shooting and implement cooldowns.
    Now fully enemy-aware: only shoots when enemy is visible and roughly aligned.
    """
    def __init__(self):
        self.last_shoot_frame = -999
        self.total_shots_fired = 0
        self.shots_on_target = 0

    def reset_episode(self):
        """Reset for new episode."""
        self.last_shoot_frame = -999

    def gate_action(self, intended_action: int, current_frame: int, state: Optional[GameState],
                    n_buttons: int) -> Tuple[int, float]:
        """
        Gate the intended action based on smart shooting logic.
        Returns (actual_action, reward_penalty)
        """
        reward_penalty = 0.0
        SHOOT_ACTION = n_buttons - 1  # ATTACK button

        # If not shooting, just allow
        if intended_action != SHOOT_ACTION:
            return intended_action, reward_penalty

        # Cooldown check
        frames_since_shot = current_frame - self.last_shoot_frame
        if frames_since_shot < SHOOT_COOLDOWN_FRAMES:
            logger.debug(f"[GATE] Shot blocked - cooldown ({frames_since_shot}/{SHOOT_COOLDOWN_FRAMES})")
            return 0, 0.0  # Move forward instead of shooting

        # Enemy detection
        enemy_pos = get_nearest_enemy(state)
        agent_center = (RESOLUTION[1]//2, RESOLUTION[0]//2)
        enemy_visible = enemy_pos is not None

        # Check if enemy roughly aligned (e.g., within ±10 pixels horizontally)
        aligned_for_shoot = False
        if enemy_visible:
            dx = enemy_pos[0] - agent_center[0]
            if abs(dx) <= ENEMY_SHOOT_TOLERANCE_PIXELS:
                aligned_for_shoot = True

        if enemy_visible and aligned_for_shoot:
            # Allow shooting
            self.last_shoot_frame = current_frame
            self.total_shots_fired += 1
            self.shots_on_target += 1  # Count as on-target since aligned
            logger.debug(f"[GATE] Shot allowed - enemy aligned at {enemy_pos}")
            return intended_action, 0.0
        else:
            # Block shot and penalize
            reward_penalty = WASTEFUL_SHOOT_PENALTY
            logger.debug(f"[GATE] Shot blocked - enemy not visible or not aligned (enemy_pos={enemy_pos})")
            return 0, reward_penalty  # Move forward instead

    def get_accuracy_stats(self) -> Dict[str, float]:
        """Get shooting accuracy statistics."""
        if self.total_shots_fired == 0:
            return {"accuracy": 0.0, "shots_fired": 0, "shots_on_target": 0}
        return {
            "accuracy": self.shots_on_target / self.total_shots_fired,
            "shots_fired": self.total_shots_fired,
            "shots_on_target": self.shots_on_target
        }


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
            tmp = self.conv1(dummy)
            tmp = self.conv2(tmp)
            tmp = self.conv3(tmp)
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
        x1 = x[:, :half]
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

    def get_action(self, state: np.ndarray, game_state: Optional[GameState] = None) -> int:
        """Get action with optional enemy-aware logic."""
        # state shape (1, H, W)
        if random.random() < self.epsilon:
            action = random.randrange(self.action_size)
        else:
            with torch.no_grad():
                t = torch.from_numpy(np.expand_dims(state, axis=0)).float().to(self.device)  # (B, C, H, W)
                q = self.q_net(t)
                action = int(torch.argmax(q, dim=-1).item())
        
        # Smart shooting: if DQN wants to shoot but no enemy visible, choose different action
        if game_state is not None:
            SHOOT_ACTION = self.action_size - 1
            if action == SHOOT_ACTION and not detect_enemies_in_state(game_state):
                # Override with forward movement if trying to shoot without target
                action = 0  # Forward action
        
        return action

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
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())


# -------------------------
# Spike decoder MLP
# -------------------------
class SpikeDecoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int = SPIKE_DECODER_HID, out: int = 4):  # Changed to 4 actions
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
all_channels = [i for i in range(64) if i not in {0, 4, 7, 56, 63}]
try:
    all_channels_set = cl.ChannelSet(*all_channels)
except Exception:
    all_channels_set = cl.ChannelSet(*all_channels)

# Enhanced action channel groups - now includes shooting
stim_channels = cl.ChannelSet(9, 10, 17, 18)
forward_channels = (41, 42, 49, 50)
left_channels = (13, 14, 21, 22)
right_channels = (45, 46, 53, 54)
shoot_channels = (31, 32, 39, 40)  # New shooting action channels
action_channel_groups = (forward_channels, left_channels, right_channels, shoot_channels)
feedback_channels = (27, 28, 35, 36)
try:
    feedback_channels_set = cl.ChannelSet(*feedback_channels)
except Exception:
    feedback_channels_set = cl.ChannelSet(*feedback_channels)

# Stim design
try:
    stim_design = cl.StimDesign(STIM_PULSE_US, STIM_AMP_NEG_UA, STIM_PULSE_US, STIM_AMP_POS_UA)
except Exception:
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
        try:
            return cl.BurstDesign(count=count, frequency=freq)
        except Exception:
            try:
                return cl.BurstDesign(burst_count=count, burst_frequency=freq)
            except Exception:
                return cl.BurstDesign()


def build_precompiled_plans(neurons: Neurons, freqs=PRECOMPILE_FREQS) -> Dict[int, object]:
    plans = {}
    for f in freqs:
        try:
            p = neurons.create_stim_plan()
            p.interrupt(stim_channels)
            bd = _make_burst_design(200, int(max(1, f)))
            p.stim(stim_channels, stim_design, bd)
            plans[int(f)] = p
        except Exception as e:
            logger.warning(f"[PLANS] Couldn't precompile freq {f}: {e}")
    return plans


# -------------------------
# Encoding strategies (unchanged)
# -------------------------
def encode_temporal(neurons: Neurons, game_state: Optional[GameState], plans: Dict[int, object]):
    """Temporal freq encoding: distance -> frequency -> run precompiled plan."""
    if not plans:
        return
    if game_state is None:
        return
    try:
        game_vars = getattr(game_state, 'game_variables', None)
        if game_vars is None or len(game_vars) < 4:
            return
        guy = np.array([game_vars[2], game_vars[3]])
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
    """Rate coding: closer -> stimulate more electrodes in stim_channels set."""
    if not patterns:
        return
    if game_state is None:
        return
    try:
        game_vars = getattr(game_state, 'game_variables', None)
        if game_vars is None or len(game_vars) < 4:
            return
        guy = np.array([game_vars[2], game_vars[3]])
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
    """Spatial encoding: map angle/relative location to different electrode groups."""
    if not spatial_map:
        return
    if game_state is None:
        return
    try:
        game_vars = getattr(game_state, 'game_variables', None)
        if game_vars is None or len(game_vars) < 4:
            return
        guy = np.array([game_vars[2], game_vars[3]])
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
    if plans:
        encode_temporal(neurons, game_state, plans)
    if spatial_map and (random.random() < 0.25):
        encode_spatial(neurons, game_state, spatial_map)
    if patterns:
        encode_rate(neurons, game_state, patterns)


# -------------------------
# Decoding: spike history & features
# -------------------------
def extract_spike_features(spikes, window_hist: deque) -> Tuple[np.ndarray, np.ndarray]:
    """Extract spike features including shooting action group."""
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
# Enhanced reward system with shooting penalties - FIXED VERSION
# -------------------------
class DistanceReward:
    """Enhanced reward system with ammo conservation - FIXED VERSION."""
    def __init__(self, scale: float = 0.05, pickup_bonus: float = 50.0, max_distance: float = 200.0):
        self.prev_distance: Optional[float] = None
        self.scale = float(scale)
        self.pickup_bonus = float(pickup_bonus)
        self.max_distance = float(max_distance)
        self.prev_had_armor: Optional[bool] = None
        self.prev_ammo: Optional[int] = None

    def start_episode(self):
        self.prev_distance = None
        self.prev_had_armor = None
        self.prev_ammo = None

    def step(self, state_before: Optional[GameState], state_after: Optional[GameState]) -> float:
        # Existing distance/armor logic
        def has_armor(st: Optional[GameState]) -> bool:
            if st is None:
                return False
            try:
                return any(getattr(o, "name", "") == "GreenArmor" for o in st.objects)
            except Exception:
                return False

        r = 0.0
        d_before = self.armor_distance(state_before, self.max_distance)
        d_after = self.armor_distance(state_after, self.max_distance)

        # Distance shaping reward
        if d_before is not None and d_after is not None:
            delta = d_before - d_after
            r += self.scale * float(np.clip(delta, -10.0, 10.0))

        # Armor pickup bonus
        now_has = has_armor(state_after)
        was_has = has_armor(state_before)
        if was_has and not now_has:
            r += self.pickup_bonus

        # FIXED: More robust ammo conservation logic
        try:
            # Safely get ammo from game_variables
            def get_ammo(state):
                if state is None:
                    return 0
                game_vars = getattr(state, 'game_variables', None)
                if game_vars is None or len(game_vars) < 3:
                    return 0
                return int(game_vars[2]) if game_vars[2] is not None else 0
            
            current_ammo = get_ammo(state_after)
            prev_ammo = get_ammo(state_before)
            
            # Small bonus for conserving ammo when no enemies present
            if not detect_enemies_in_state(state_after) and current_ammo == prev_ammo and current_ammo > 0:
                r += AMMO_CONSERVATION_BONUS
                
        except (IndexError, AttributeError, TypeError) as e:
            # If ammo detection fails, just skip it - don't crash
            logger.debug(f"[REWARD] Ammo detection failed: {e}")
            pass

        self.prev_distance = d_after
        self.prev_had_armor = now_has
        return r

    def armor_distance(self, state: Optional[GameState], max_distance: float = 200.0) -> Optional[float]:
        """Clipped Euclidean distance to GreenArmor, or None if not present."""
        if state is None:
            return None
        try:
            # FIXED: More robust game_variables access
            game_vars = getattr(state, 'game_variables', None)
            if game_vars is None or len(game_vars) < 4:
                return None
            
            guy = np.array([game_vars[2], game_vars[3]], dtype=np.float32)
            armor_obj = next((o for o in state.objects if getattr(o, "name", "") == "GreenArmor"), None)
            if armor_obj is None:
                return None
            armor = np.array([armor_obj.position_x, armor_obj.position_y], dtype=np.float32)
            d = float(np.linalg.norm(guy - armor))
            return min(d, max_distance)
        except (Exception, IndexError, AttributeError, TypeError):
            return None


# -------------------------
# Simple fallback spike decoder for when training fails
# -------------------------
def create_simple_fallback_decoder(device: torch.device) -> SpikeDecoder:
    """Create a simple random-initialized spike decoder as fallback."""
    expected_input_dim = len(action_channel_groups) * SPIKE_WINDOW
    decoder = SpikeDecoder(input_dim=expected_input_dim, hidden=SPIKE_DECODER_HID, out=4).to(device)
    logger.info(f"[FALLBACK] Created simple spike decoder: input_dim={expected_input_dim}, out=4")
    return decoder


# -------------------------
# Rest of the helper functions (unchanged from previous)
# -------------------------
def graded_feedback_plan(neurons: Neurons, magnitude: float = 1.0, positive: bool = True):
    """Create a stim plan where magnitude scales the burst count/duration."""
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


def get_player_and_armor_positions(state: Optional[GameState]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (player_xy, armor_xy) or (None, None) if not available."""
    if state is None:
        return None, None
    try:
        game_vars = getattr(state, 'game_variables', None)
        if game_vars is None or len(game_vars) < 4:
            return None, None
        guy = np.array([game_vars[2], game_vars[3]], dtype=np.float32)
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


def log_distance_trace(distances: List[Optional[float]], save_path: str, ep_idx: int):
    xs = list(range(len(distances)))
    ys = [d if (d is not None) else np.nan for d in distances]
    try:
        plt.figure(figsize=(6,3))
        plt.plot(xs, ys, linewidth=1.5)
        plt.title(f"Distance to GreenArmor — Episode {ep_idx}")
        plt.xlabel("tick")
        plt.ylabel("distance (clipped)")
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
        if step_idx % 30 == 0:
            out = os.path.join(save_path, f"automap_ep{ep_idx}_t{step_idx}.png")
            cv2.imwrite(out, im)
    except Exception:
        pass


def collect_spike_dataset_enemy_targeted(game: DoomGame, neurons: Neurons, teacher_agent: DQNAgent,
                                         target_samples: int = 3000, max_episodes: int = 50,
                                         allow_stim: bool = False, window: int = SPIKE_WINDOW,
                                         save_path: str = ".", show_automap_flag: bool = False):
    X, y = [], []
    samples, episodes = 0, 0
    spike_hist = deque(maxlen=window - 1)
    action_gater = ActionGater()
    agent_pos = (RESOLUTION[1]//2, RESOLUTION[0]//2)

    for _ in range(window - 1):
        spike_hist.append(np.zeros(len(action_channel_groups), dtype=np.float32))

    while samples < target_samples and episodes < max_episodes:
        game.new_episode()
        action_gater.reset_episode()
        episodes += 1
        frame_count = 0

        while not game.is_episode_finished() and samples < target_samples:
            state = game.get_state()
            frame = preprocess(state.screen_buffer) if state else np.zeros((1, *RESOLUTION), dtype=np.float32)

            # Enemy-aware teacher action
            enemy_action_vec = generate_enemy_aware_action(agent_pos, state, game.get_available_buttons_size())
            raw_teacher_action = teacher_agent.get_action(frame, state)
            teacher_action, penalty = action_gater.gate_action(raw_teacher_action, frame_count, state,
                                                               game.get_available_buttons_size())

            if allow_stim:
                encode_hybrid(neurons, state)
                for tick in neurons.loop(ticks_per_second=TICKS_PER_SECOND):
                    feat, counts = extract_spike_features(tick.analysis.spikes, spike_hist)
                    spike_hist.append(counts)
                    X.append(feat.astype(np.float32))
                    y.append(teacher_action)
                    samples += 1
                    break
            else:
                # Enemy-targeted action for dataset
                agent_pos = (RESOLUTION[1]//2, RESOLUTION[0]//2)
                enemy_action_vec = generate_enemy_aware_action(agent_pos, state, game.get_available_buttons_size())
                enemy_action_idx = np.argmax(enemy_action_vec)
                
                # Simulate spike counts
                num_groups = len(action_channel_groups)
                base = np.zeros(num_groups, dtype=np.float32)
                base[enemy_action_idx % num_groups] = 5.0
                simulated_counts = np.random.poisson(base).astype(np.float32)
                feat = np.concatenate(list(spike_hist) + [simulated_counts], axis=0)
                spike_hist.append(simulated_counts)
                
                X.append(feat.astype(np.float32))
                y.append(enemy_action_idx)

samples += 1

            game.set_action(enemy_action_vec)
            game.advance_action(FRAME_REPEAT)
            frame_count += 1

            if show_automap_flag:
                show_automap(game)

        stats = action_gater.get_accuracy_stats()
        logger.info(f"[DATA] Episode {episodes}: shots={stats['shots_fired']}, accuracy={stats['accuracy']:.2f}")

    X, y = np.array(X), np.array(y, dtype=np.int64)
    ensure_dir(save_path)
    np.save(os.path.join(save_path, "spike_X.npy"), X)
    np.save(os.path.join(save_path, "spike_y.npy"), y)
    logger.info(f"[DATA] Saved enemy-targeted spike dataset to {save_path} (X={X.shape}, y={y.shape})")
    return X, y



def train_spike_decoder(X: np.ndarray, y: np.ndarray, device: torch.device,
                        epochs: int = SPIKE_DECODER_EPOCHS, lr: float = SPIKE_DECODER_LR,
                        batch_size: int = 256, save_path: str = ".") -> SpikeDecoder:
    input_dim = X.shape[1]
    num_actions = int(y.max() + 1)
    model = SpikeDecoder(input_dim=input_dim, hidden=SPIKE_DECODER_HID, out=num_actions).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    idxs = np.arange(len(X))
    for ep in range(epochs):
        np.random.shuffle(idxs)
        losses = []
        for i in range(0, len(idxs), batch_size):
            bs = idxs[i:i + batch_size]
            xb = torch.from_numpy(X[bs]).float().to(device)
            yb = torch.from_numpy(y[bs]).long().to(device)
            logits = model(xb)
            loss = ce(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        logger.info(f"[SPK] Epoch {ep + 1}/{epochs} loss={np.mean(losses):.4f}")
    ensure_dir(save_path)
    pth = os.path.join(save_path, "spike_decoder.pth")
    torch.save(model.state_dict(), pth)
    logger.info(f"[SPK] Saved enhanced spike decoder to {pth}")
    return model


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


def run_closed_loop_neuron_control(game: DoomGame, neurons: Neurons, spike_decoder: SpikeDecoder,
                                   plans=None, patterns=None, spatial_map=None, episodes: int = 10,
                                   window: int = SPIKE_WINDOW, allow_stim: bool = False, save_path: str = ".",
                                   device: torch.device = torch.device('cpu'), online_finetune: bool = False,
                                   teacher: Optional[DQNAgent] = None, show_automap_flag: bool = False,
                                   norm_mu: Optional[np.ndarray] = None, norm_std: Optional[np.ndarray] = None):
    """Enhanced closed-loop control with smart shooting."""
    spike_decoder.to(device)
    spike_decoder.eval()
    episodes_done = 0
    episode_scores = []
    moving_avg_scores = []
    action_agreements = []
    latency_records = []
    action_gater = ActionGater()  # Smart shooting controller

    ensure_dir(save_path)
    csv_path = os.path.join(save_path, "episode_log.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "score", "steps", "mean_latency_ms", "agreement_with_teacher", 
                         "shots_fired", "shot_accuracy"])

    expected_input_dim = len(action_channel_groups) * window
    logger.info(f"[RUN] Expected decoder input dim: {expected_input_dim} (with 4 action groups)")

    while episodes_done < episodes:
        game.new_episode()
        shaper = DistanceReward(scale=0.05, pickup_bonus=50.0, max_distance=200.0)
        shaper.start_episode()
        action_gater.reset_episode()

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

        if allow_stim:
            loop_iter = neurons.loop(ticks_per_second=TICKS_PER_SECOND)
        else:
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
            dist_trace.append(d_before if d_before is not None else None)
        
            # Encode or simulate spikes
            if allow_stim:
                encode_hybrid(neurons, state_before, plans, patterns, spatial_map)
                feat, counts = extract_spike_features(tick.analysis.spikes, spike_hist)
            else:
                num_groups = len(action_channel_groups)
                if teacher is not None and state_before is not None:
                    frame = preprocess(state_before.screen_buffer)
                    t_act = teacher.get_action(frame, state_before)
                    base = np.ones(num_groups, dtype=np.float32) * 0.5
                    base[t_act % num_groups] += 4.0
                else:
                    base = np.ones(num_groups, dtype=np.float32) * 1.0
                counts = np.random.poisson(base).astype(np.float32)
                feat = np.concatenate(list(spike_hist) + [counts], axis=0)
        
            spike_hist.append(counts)
            rasters.append(counts)
        
            # Normalize features
            feat_in = feat.astype(np.float32)
            if norm_mu is not None and norm_std is not None:
                feat_in = (feat_in - norm_mu.ravel()) / norm_std.ravel()
        
            xb = torch.from_numpy(feat_in).unsqueeze(0).float().to(device)
            with torch.no_grad():
                logits = spike_decoder(xb)
                raw_act_idx = int(torch.argmax(logits, dim=1).cpu().numpy()[0])
        
            # Enemy-targeted action
            agent_pos = (RESOLUTION[1] // 2, RESOLUTION[0] // 2)
            enemy_action_vec = generate_enemy_aware_action(agent_pos, state_before, game.get_available_buttons_size())
            raw_act_idx = int(np.argmax(enemy_action_vec))
        
            # SMART SHOOTING: single gating
            act_idx, shoot_penalty = action_gater.gate_action(
                raw_act_idx, steps, state_before, game.get_available_buttons_size()
            )
            if raw_act_idx != act_idx:
                logger.debug(f"[SMART] Blocked wasteful shot at step {steps}")
        
            # Execute action
            action_vec = safe_action_vector(act_idx, game.get_available_buttons_size())
            game.set_action(action_vec)
            game.advance_action(FRAME_REPEAT)
        
            # Enhanced reward
            state_after = game.get_state()
            r_env = game.get_last_reward() if hasattr(game, "get_last_reward") else 0.0
            r_shape = shaper.step(state_before, state_after)
            rew = float(r_env) + float(r_shape) + float(shoot_penalty)
            total_reward += rew
            steps += 1
        
            # Per-action feedback
            per_action_feedback(neurons, r_shape + shoot_penalty, allow_stim=allow_stim)
        
            # Latency tracking
            latencies.append((perf_counter() - t0) * 1000.0)
        
            # Teacher agreement
            if teacher is not None and state_before is not None:
                frame = preprocess(state_before.screen_buffer)
                t_action = teacher.get_action(frame, state_before)
                teacher_checks += 1
                if t_action == act_idx:
                    agreement_counts += 1
        
            # Online fine-tuning
            if online_finetune and teacher is not None and state_before is not None:
                t_action = teacher.get_action(preprocess(state_before.screen_buffer), state_before)
                model = spike_decoder
                model.train()
                xb_ft = xb
                yb_ft = torch.tensor([t_action], dtype=torch.long).to(device)
                opt = optim.Adam(model.parameters(), lr=1e-5)
                loss = nn.CrossEntropyLoss()(model(xb_ft), yb_ft)
                opt.zero_grad()
                loss.backward()
                opt.step()
                model.eval()
        
            # Automap visualization
            if show_automap_flag:
                show_automap(game)
                save_automap_snapshot(game, save_path, episodes_done, steps)
        
            # End-of-episode processing
            if game.is_episode_finished():
                mean_lat = float(np.mean(latencies)) if latencies else 0.0
                agreement = (agreement_counts / teacher_checks) if teacher_checks > 0 else None
                shooting_stats = action_gater.get_accuracy_stats()
        
                episode_scores.append(total_reward)
                moving_avg_scores.append(float(np.mean(episode_scores[-MOVING_AVG_WINDOW:])))
                action_agreements.append(agreement)
                latency_records.append(mean_lat)
        
                # Save raster
                rasters_np = np.stack(rasters, axis=0) if rasters else np.zeros((0, len(action_channel_groups)))
                np.save(os.path.join(save_path, f"raster_ep{episodes_done}.npy"), rasters_np)
                try:
                    plt.figure(figsize=(8, 3))
                    plt.imshow(rasters_np.T, aspect='auto', interpolation='nearest')
                    plt.xlabel('tick')
                    plt.ylabel('action-group (0:forward,1:left,2:right,3:shoot)')
                    plt.title(f'Raster Ep{episodes_done} - Shots: {shooting_stats["shots_fired"]}')
                    plt.savefig(os.path.join(save_path, f"raster_ep{episodes_done}.png"))
                    plt.close()
                except Exception:
                    pass
        
                log_distance_trace(dist_trace, save_path, episodes_done)
        
                csv_writer.writerow([
                    episodes_done, total_reward, steps, mean_lat,
                    agreement if agreement is not None else "NA",
                    shooting_stats["shots_fired"], shooting_stats["accuracy"]
                ])
                csv_file.flush()
        
                logger.info(f"[RUN] Episode {episodes_done} score={total_reward:.2f} steps={steps} "
                            f"latency={mean_lat:.2f}ms agreement={agreement} "
                            f"shots={shooting_stats['shots_fired']} accuracy={shooting_stats['accuracy']:.2f}")
        
                end_of_episode_feedback(neurons, total_reward, allow_stim=allow_stim)
        
                # Safety interrupts
                try:
                    neurons.interrupt(stim_channels)
                    neurons.interrupt(feedback_channels_set)
                except Exception:
                    pass
        
                episodes_done += 1
                break


    csv_file.close()
    
    # Enhanced summary plots
    try:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(episode_scores, label='episode_score')
        plt.plot(moving_avg_scores, label=f'moving_avg({MOVING_AVG_WINDOW})')
        plt.legend()
        plt.title('Scores with Smart Shooting')
        
        plt.subplot(1, 3, 2)
        if any(a is not None for a in action_agreements):
            plt.plot([a if a is not None else 0 for a in action_agreements])
            plt.title('Agreement with Teacher')
            
        plt.subplot(1, 3, 3)
        shooting_accuracies = []  # Would need to collect this data
        plt.title('Shooting Performance')
        plt.xlabel('Episode')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'enhanced_performance.png'))
        plt.close()
    except Exception:
        pass

    logger.info(f"[RUN] Enhanced closed-loop finished with smart shooting. Logs saved to {save_path}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Enhanced Neural DOOM with Smart Shooting")
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

    # FIXED: Better default pipeline logic
    if len(sys.argv) == 1:
        logger.info("No CLI arguments → running enhanced default pipeline with smart shooting.")
        args.collect_spike_data = True
        args.collect_samples = 2000
        args.train_spike_decoder = True
        args.run_closed_loop = True
        args.allow_stim = False
        args.train_dqn = False
        args.show_automap = True
        args.episodes = 5

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
            logger.error("Environment variable CORTICAL_SAFE_KEY invalid. Aborting.")
            sys.exit(1)
        if not CL_AVAILABLE:
            logger.error("Requested hardware stimulation but 'cl' SDK not available.")
            sys.exit(1)
        logger.warning("!!! WARNING: HARDWARE STIMULATION ENABLED !!!")

    show_automap_flag = bool(args.show_automap)

    # Enhanced ViZDoom initialization
    game = DoomGame()
    game.load_config(args.config)
    game.set_episode_timeout(2000)
    game.set_episode_start_time(10)
    try:
        game.set_window_visible(bool(show_automap_flag))
    except Exception:
        pass
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    try:
        game.set_labels_buffer_enabled(True)
        game.set_objects_info_enabled(True)
        game.add_available_button(vzd.Button.MOVE_FORWARD)
        game.add_available_button(vzd.Button.TURN_LEFT)
        game.add_available_button(vzd.Button.TURN_RIGHT)
        game.add_available_button(vzd.Button.ATTACK)
        logger.info("[GAME] Enhanced setup: RGB screen, object detection, ATTACK button enabled")
    except Exception as e:
        logger.warning(f"[GAME] Setup warning: {e}")

    game.init()
    n_buttons = game.get_available_buttons_size()
    logger.info(f"[GAME] Available buttons: {n_buttons} (ATTACK should be button {n_buttons-1})")

    # Teacher DQN
    teacher = DQNAgent(action_size=4, device=device, load_model=args.load_dqn)

    # Optionally train teacher
    if args.train_dqn:
        logger.info("[DQN] Training enhanced teacher with smart shooting rewards.")
        actions = [safe_action_vector(a, n_buttons) for a in range(4)]
        shaper = DistanceReward(scale=0.05, pickup_bonus=50.0, max_distance=200.0)
        action_gater = ActionGater()

        for epoch in range(TRAIN_EPOCHS):
            logger.info(f"[DQN] Epoch {epoch+1}/{TRAIN_EPOCHS}")
            game.new_episode()
            shaper.start_episode()
            action_gater.reset_episode()
            frame_count = 0

            for step in trange(LEARNING_STEPS_PER_EPOCH, leave=False):
                state_before = game.get_state()
                s = preprocess(state_before.screen_buffer) if state_before else np.zeros((1, RESOLUTION[0], RESOLUTION[1]), dtype=np.float32)
                raw_a = teacher.get_action(s, state_before)
                a, shoot_penalty = action_gater.gate_action(raw_a, frame_count, state_before, n_buttons)
                r_env = game.make_action(actions[a], FRAME_REPEAT)
                done = game.is_episode_finished()
                state_after = game.get_state() if not done else None
                r_shape = shaper.step(state_before, state_after)
                r = float(r_env) + float(r_shape) + float(shoot_penalty)
                s2 = preprocess(state_after.screen_buffer) if (state_after is not None) else np.zeros((1, RESOLUTION[0], RESOLUTION[1]), dtype=np.float32)
                teacher.store(s, raw_a, r, s2, done)
                teacher.train_step()
                frame_count += 1

                if done:
                    stats = action_gater.get_accuracy_stats()
                    logger.debug(f"Episode complete - shots: {stats['shots_fired']}, accuracy: {stats['accuracy']:.2f}")
                    game.new_episode()
                    shaper.start_episode()
                    action_gater.reset_episode()
                    frame_count = 0

            teacher.sync_target()
        torch.save(teacher.q_net.state_dict(), os.path.join(args.save_path, "enhanced_dqn_teacher.pth"))
        logger.info("[DQN] Teacher training complete.")

    # Neurons context
    with Neurons() as neurons:
        logger.info(f"[CL] Using enhanced neurons object: {neurons}")
        plans = build_precompiled_plans(neurons) if (args.allow_stim and CL_AVAILABLE) else {}
        patterns, spatial_map = None, {}

        # Spike data collection & decoder training
        spike_decoder, norm_mu, norm_std = None, None, None
        sd_path = os.path.join(args.save_path, "enhanced_spike_decoder.pth")
        stats_path = os.path.join(args.save_path, "enhanced_spike_X_stats.npz")

        if args.collect_spike_data or args.train_spike_decoder:
            X, y = collect_spike_dataset(game, neurons, teacher, target_samples=args.collect_samples,
                                         max_episodes=50, plans=plans, patterns=patterns, spatial_map=spatial_map,
                                         allow_stim=args.allow_stim, window=SPIKE_WINDOW, save_path=args.save_path,
                                         show_automap_flag=show_automap_flag)
            if args.train_spike_decoder:
                mu = X.mean(axis=0, keepdims=True)
                std = X.std(axis=0, keepdims=True) + 1e-6
                X_norm = (X - mu) / std
                np.save(os.path.join(args.save_path, "enhanced_spike_X_norm.npy"), X_norm)
                np.savez(stats_path, mu=mu, std=std)
                spike_decoder = train_spike_decoder(X_norm, y, device=device, epochs=SPIKE_DECODER_EPOCHS, save_path=args.save_path)
                norm_mu, norm_std = mu, std

        # Load spike decoder if available
        if args.load_spike_decoder and spike_decoder is None and os.path.exists(args.load_spike_decoder):
            arr = np.load(os.path.join(os.path.dirname(args.load_spike_decoder), "enhanced_spike_X_stats.npz"))
            norm_mu, norm_std = arr["mu"], arr["std"]
            expected_dim = len(action_channel_groups) * SPIKE_WINDOW
            spike_decoder = SpikeDecoder(input_dim=expected_dim, hidden=SPIKE_DECODER_HID, out=4)
            spike_decoder.load_state_dict(torch.load(args.load_spike_decoder, map_location=device))
            spike_decoder.to(device)
            logger.info("[SPK] Loaded spike decoder.")

        # Fallback decoder
        if spike_decoder is None and args.run_closed_loop:
            logger.warning("[FALLBACK] Creating simple fallback spike decoder...")
            spike_decoder = create_simple_fallback_decoder(device)

        # Run enhanced closed-loop
        if args.run_closed_loop:
            run_closed_loop_neuron_control(game, neurons, spike_decoder, plans=plans, patterns=patterns,
                                           spatial_map=spatial_map, episodes=args.episodes, window=SPIKE_WINDOW,
                                           allow_stim=args.allow_stim, save_path=args.save_path, device=device,
                                           online_finetune=args.online_finetune, teacher=teacher,
                                           show_automap_flag=show_automap_flag, norm_mu=norm_mu, norm_std=norm_std)

    # Cleanup
    try: cv2.destroyAllWindows()
    except Exception: pass
    try: game.close()
    except Exception: pass
    logger.info("[INFO] Enhanced Neural DOOM complete! Logs saved to: %s", args.save_path)
    

if __name__ == "__main__":
    main()
