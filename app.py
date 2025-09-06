```python
#!/usr/bin/env python3
import sys
"""
Hybrid closed-loop ViZDoom + cl (neural stim) system — improved.

Improvements added:
 - Advanced encoding (temporal / rate / spatial / hybrid)
 - Precompiled stim plans for common frequencies/amplitudes
 - Graded feedback (multiple magnitudes/durations)
 - Spike-history features + SpikeDecoder (MLP) with offline & optional online fine-tune
 - Extensive logging: CSV episode logs, spike rasters (.npy), action agreement plots
 - Latency monitoring and low-jitter loop via neurons.loop()
 - Simulation-only default; explicit --allow-stim required to perform hardware stimulation

SAFETY: Stimulating biological tissue is potentially harmful. Do not run with --allow-stim on humans/animals
without approvals and trained staff. Default mode is safe simulation.
"""

import os
import argparse
import random
import csv
from collections import deque
from time import time

import numpy as np
import skimage.transform
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

import vizdoom as vzd
from vizdoom import DoomGame, GameState, Mode

import cl  # cl SDK
from cl import Neurons
from cl.closed_loop import LoopTick
from cl.stim_plan import StimPlan

import cv2

# -------------------------
# Config / Hyperparameters
# -------------------------
RESOLUTION = (30, 45)         # Preprocessing for DQN teacher
FRAME_REPEAT = 1
TICKS_PER_SECOND = 35

# DQN teacher-related (kept lightweight - adapt as needed)
LEARNING_RATE = 0.00025
DISCOUNT_FACTOR = 0.99
TRAIN_EPOCHS = 3               # shorten default for speed, increase as needed
LEARNING_STEPS_PER_EPOCH = 1500
REPLAY_MEMORY_SIZE = 10000
BATCH_SIZE = 64

# Spike dataset & decoder
SPIKE_WINDOW = 8               # longer history for improved decoding
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

# Logging
MOVING_AVG_WINDOW = 10

# Reproducibility
SEED = 1234

# -------------------------
# Utilities
# -------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def preprocess(img):
    """Convert VizDoom screen buffer to grayscale RESOLUTION (1,H,W)."""
    if img is None:
        return np.zeros((1, RESOLUTION[0], RESOLUTION[1]), dtype=np.float32)
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 3 and img.shape[2] == 3:
        gray = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    else:
        gray = img if img.ndim == 2 else np.mean(img, axis=0)
    resized = skimage.transform.resize(gray, RESOLUTION, anti_aliasing=True)
    return np.expand_dims(resized.astype(np.float32), axis=0)

# -----------------------------
# Optional automap visualization
# -----------------------------
def show_automap(game, sleep_ms=28):
    """
    Display the automap in a CV2 window. Non-blocking.
    Call this inside your main loop tick if --show-automap is enabled.
    """
    state = game.get_state()
    if state is None:
        return
    automap = state.automap_buffer
    if automap is not None:
        cv2.imshow("ViZDoom Automap", automap)
        cv2.waitKey(sleep_ms)  # short delay for rendering

# -------------------------
# Dueling DQN (teacher)
# -------------------------
class DuelQNet(nn.Module):
    def __init__(self, available_actions_count):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 8, kernel_size=3, stride=2, bias=False),
                                   nn.BatchNorm2d(8), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(8, 8, kernel_size=3, stride=2, bias=False),
                                   nn.BatchNorm2d(8), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False),
                                   nn.BatchNorm2d(8), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=False),
                                   nn.BatchNorm2d(16), nn.ReLU())
        self.state_fc = nn.Sequential(nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage_fc = nn.Sequential(nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, available_actions_count))
    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x); x = self.conv4(x)
        x = x.view(-1, 192); x1 = x[:, :96]; x2 = x[:, 96:]
        state_value = self.state_fc(x1).reshape(-1, 1)
        advantage_values = self.advantage_fc(x2)
        return state_value + (advantage_values - advantage_values.mean(dim=1, keepdim=True))

class DQNAgent:
    def __init__(self, action_size, device, load_model=None):
        self.action_size = action_size
        self.device = device
        self.q_net = DuelQNet(action_size).to(device)
        self.target_net = DuelQNet(action_size).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.opt = optim.SGD(self.q_net.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.batch_size = BATCH_SIZE
        self.discount = DISCOUNT_FACTOR
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        if load_model and os.path.exists(load_model):
            self.q_net.load_state_dict(torch.load(load_model, map_location=device))
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.epsilon = self.epsilon_min

    def get_action(self, state):
        # state shape (1,H,W)
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            t = torch.from_numpy(np.expand_dims(state, axis=0)).float().to(self.device)
            q = self.q_net(t)
            return int(torch.argmax(q).cpu().numpy()[0])

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
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

# -------------------------
# Spike decoder MLP
# -------------------------
class SpikeDecoder(nn.Module):
    def __init__(self, input_dim, hidden=SPIKE_DECODER_HID, out=3):
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
all_channels = [i for i in range(64) if i not in {0,4,7,56,63}]
all_channels_set = cl.ChannelSet(*all_channels)

stim_channels = cl.ChannelSet(9, 10, 17, 18)   # encoding channels (can expand)
forward_channels = (41,42,49,50)
left_channels = (13,14,21,22)
right_channels = (45,46,53,54)
action_channel_groups = (forward_channels, left_channels, right_channels)
feedback_channels = (27,28,35,36)
feedback_channels_set = cl.ChannelSet(*feedback_channels)

stim_design = cl.StimDesign(STIM_PULSE_US, STIM_AMP_NEG_UA, STIM_PULSE_US, STIM_AMP_POS_UA)

# Precompile stim plans for each frequency to avoid per-tick plan creation
def build_precompiled_plans(neurons: Neurons, freqs=PRECOMPILE_FREQS):
    plans = {}
    for f in freqs:
        p = neurons.create_stim_plan()
        # interrupt only the stim_channels before running this plan to avoid global jitter
        p.interrupt(stim_channels)
        # moderate burst length; user can override
        p.stim(stim_channels, stim_design, cl.BurstDesign(200, int(max(1, f))))
        plans[f] = p
    return plans

# -------------------------
# Encoding strategies
# -------------------------
def encode_temporal(neurons: Neurons, game_state: GameState, plans: dict):
    """Temporal freq encoding: distance -> frequency -> run precompiled plan"""
    if game_state is None or len(game_state.objects) < 2:
        return
    try:
        guy = np.array(game_state.game_variables[2:4])
        armor_obj = next(obj for obj in game_state.objects if obj.name == "GreenArmor")
        armor = np.array([armor_obj.position_x, armor_obj.position_y])
        d = np.linalg.norm(guy - armor)
    except Exception:
        return
    max_d = 200.0
    d = min(d, max_d)
    min_f, max_f = 4, 40
    f = int(round(max_f - (d/max_d) * (max_f - min_f)))
    # snap to nearest precompiled freq
    freqs = sorted(plans.keys())
    chosen = min(freqs, key=lambda x: abs(x - f))
    plans[chosen].run()   # run precompiled plan (low latency)

def encode_rate(neurons: Neurons, game_state: GameState, patterns=None):
    """
    Rate coding: closer -> stimulate more electrodes in stim_channels set (spatial recruitment).
    'patterns' is a list of ChannelSets pre-defined from 1..N electrodes.
    """
    if game_state is None or len(game_state.objects) < 2:
        return
    try:
        guy = np.array(game_state.game_variables[2:4])
        armor_obj = next(obj for obj in game_state.objects if obj.name == "GreenArmor")
        armor = np.array([armor_obj.position_x, armor_obj.position_y])
        d = np.linalg.norm(guy - armor)
    except Exception:
        return
    max_d = 200.0
    d = min(d, max_d)
    # map to index of pattern
    ratio = 1.0 - (d / max_d)
    idx = int(round(ratio * (len(patterns)-1)))
    # interrupt only the channels about to be stimulated
    patterns[idx].interrupt(patterns[idx])
    patterns[idx].run()

def encode_spatial(neurons: Neurons, game_state: GameState, spatial_map=None):
    """
    Spatial encoding: map angle/relative location to different electrode groups.
    spatial_map: dict of sector -> ChannelSet
    """
    if game_state is None or len(game_state.objects) < 2:
        return
    try:
        guy = np.array(game_state.game_variables[2:4])
        armor_obj = next(obj for obj in game_state.objects if obj.name == "GreenArmor")
        armor = np.array([armor_obj.position_x, armor_obj.position_y])
        vec = armor - guy
        angle = np.arctan2(vec[1], vec[0])  # radians
    except Exception:
        return
    # map angle to sector 0..N-1
    sectors = sorted(spatial_map.keys())
    sector_count = len(sectors)
    sector_idx = int(((angle + np.pi) / (2*np.pi)) * sector_count) % sector_count
    chosen = sectors[sector_idx]
    plan = spatial_map[chosen]
    plan.interrupt(plan)
    plan.run()

# Hybrid encoder wrapper
def encode_hybrid(neurons: Neurons, game_state: GameState, plans=None, patterns=None, spatial_map=None):
    # combine temporal + rate + spatial with weighted choice
    encode_temporal(neurons, game_state, plans)
    # small chance to add spatial variation
    encode_spatial(neurons, game_state, spatial_map)
    # rate coding as complement
    encode_rate(neurons, game_state, patterns)

# -------------------------
# Decoding: spike history & features
# -------------------------
def extract_spike_features(spikes, window_hist):
    """
    spikes: list of spike objects for current tick
    window_hist: deque of past spike-count arrays (len = SPIKE_WINDOW-1)
    returns flattened feature vector shape (3 * SPIKE_WINDOW)
    """
    counts = np.zeros(len(action_channel_groups), dtype=np.float32)
    for sp in spikes:
        ch = sp.channel
        for i,grp in enumerate(action_channel_groups):
            if ch in grp:
                counts[i] += 1
    # append to history and create flattened vector
    hist_copy = list(window_hist) + [counts]
    feat = np.concatenate(hist_copy, axis=0)
    return feat, counts

# -------------------------
# Graded feedback (biologically inspired)
# -------------------------
def graded_feedback_plan(neurons: Neurons, magnitude=1.0, positive=True):
    """
    Create a stim plan where magnitude scales the burst count/duration.
    magnitude: 0..1 (normalized), positive True -> reward-style, False -> punishment-style
    Use less intense stimulation for small magnitudes.
    """
    plan = neurons.create_stim_plan()
    # interrupt only feedback channels
    plan.interrupt(feedback_channels_set)
    # scale burst count and frequency conservatively
    burst_count = int(5 + magnitude * 50)   # 5..55
    freq = int(80 + magnitude * 120)        # 80..200Hz
    # choose channels: positive => center set; negative => staggered channels
    if positive:
        plan.stim(feedback_channels_set, stim_design, cl.BurstDesign(burst_count, freq))
    else:
        # stimulate each feedback channel separately with slightly different bursts
        for i, ch in enumerate(feedback_channels):
            plan.stim(cl.ChannelSet(ch), stim_design, cl.BurstDesign(burst_count, freq + i*10))
    return plan

# -------------------------
# Dataset collection using teacher
# -------------------------
def collect_spike_dataset(game, neurons: Neurons, teacher_agent: DQNAgent, target_samples=3000,
                          max_episodes=50, plans=None, patterns=None, spatial_map=None,
                          allow_stim=False, window=SPIKE_WINDOW, save_path=".", show_automap_flag=False):
    """
    Runs teacher-guided episodes while encoding into neurons and collecting spikes labeled with teacher actions.
    Returns X (N, 3*window) and y (N,)
    """
    print("[DATA] Starting spike dataset collection. allow_stim=", allow_stim)
    X, y = [], []
    samples = 0; episodes = 0
    spike_hist = deque(maxlen=window-1)
    # initialize history zeros
    for _ in range(window-1):
        spike_hist.append(np.zeros(len(action_channel_groups), dtype=np.float32))

    while samples < target_samples and episodes < max_episodes:
        game.new_episode(); episodes += 1
        while not game.is_episode_finished() and samples < target_samples:
            state = game.get_state()
            frame = preprocess(state.screen_buffer) if state else np.zeros((1,RESOLUTION[0],RESOLUTION[1]), dtype=np.float32)
            teacher_action = teacher_agent.get_action(frame)
            # encode (conditionally run hardware stim if allowed)
            if allow_stim:
                encode_hybrid(neurons, state, plans, patterns, spatial_map)
            else:
                # 💡 Simulate spike counts correlated with teacher action
                num_groups = len(action_channel_groups)
                base = np.zeros(num_groups, dtype=np.float32)
                base[teacher_action % num_groups] = 5.0  # bias one group to fire more
                simulated_counts = np.random.poisson(base).astype(np.float32)

                # Build feature vector (history + new counts)
                feat = np.concatenate(list(spike_hist) + [simulated_counts], axis=0)
                spike_hist.append(simulated_counts)

                X.append(feat.astype(np.float32))
                y.append(teacher_action)
                samples += 1
                # apply teacher action to game for environment progression
                action_vec = [0]*game.get_available_buttons_size()
                if len(action_vec) >= 3: action_vec[teacher_action] = 1
                else: action_vec = [1 if i==teacher_action else 0 for i in range(len(action_vec))]
                game.set_action(action_vec); game.advance_action(FRAME_REPEAT)
                if show_automap_flag:
                    show_automap(game)
                continue  # skip neuron.loop() since we simulated

            # Use neurons.loop for a single tick to get low-jitter spike capture
            for tick in neurons.loop(ticks_per_second=TICKS_PER_SECOND):
                feat, counts = extract_spike_features(tick.analysis.spikes, spike_hist)
                spike_hist.append(counts)
                X.append(feat.astype(np.float32)); y.append(teacher_action); samples += 1
                break

            # apply teacher action to game
            action_vec = [0]*game.get_available_buttons_size()
            if len(action_vec) >= 3: action_vec[teacher_action] = 1
            else: action_vec = [1 if i==teacher_action else 0 for i in range(len(action_vec))]
            game.set_action(action_vec); game.advance_action(FRAME_REPEAT)
            if show_automap_flag:  # True if --show-automap CLI arg is set
                show_automap(game)
        print(f"[DATA] Collected {samples} samples after {episodes} episodes.")
    X = np.array(X); y = np.array(y, dtype=np.int64)
    np.save(os.path.join(save_path, "spike_X.npy"), X); np.save(os.path.join(save_path, "spike_y.npy"), y)
    return X, y

# -------------------------
# Train spike decoder
# -------------------------
def train_spike_decoder(X, y, device, epochs=SPIKE_DECODER_EPOCHS, lr=SPIKE_DECODER_LR, batch_size=256, save_path="."):
    input_dim = X.shape[1]; num_actions = int(y.max()+1)
    model = SpikeDecoder(input_dim=input_dim, hidden=SPIKE_DECODER_HID, out=num_actions).to(device)
    opt = optim.Adam(model.parameters(), lr=lr); ce = nn.CrossEntropyLoss()
    idxs = np.arange(len(X))
    for ep in range(epochs):
        np.random.shuffle(idxs); losses=[]
        for i in range(0, len(idxs), batch_size):
            bs = idxs[i:i+batch_size]
            xb = torch.from_numpy(X[bs]).float().to(device); yb = torch.from_numpy(y[bs]).long().to(device)
            logits = model(xb); loss = ce(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.item()))
        print(f"[SPK] Epoch {ep+1}/{epochs} loss={np.mean(losses):.4f}")
    torch.save(model.state_dict(), os.path.join(save_path, "spike_decoder.pth"))
    return model

# -------------------------
# Run closed-loop neuron-centric
# -------------------------
def run_closed_loop_neuron_control(game, neurons: Neurons, spike_decoder: SpikeDecoder, plans=None, patterns=None,
                                   spatial_map=None, episodes=10, window=SPIKE_WINDOW, allow_stim=False,
                                   save_path=".", device=torch.device('cpu'), online_finetune=False,
                                   teacher=None, show_automap_flag=False):
    """
    Main closed-loop evaluation where spike decoder controls game actions.
    Also logs spike rasters, action agreement with teacher (if provided), latency, and reward curves.
    """
    spike_decoder.to(device)
    spike_decoder.eval()
    episodes_done = 0
    episode_scores = []
    moving_avg_scores = []
    action_agreements = []  # fraction agreement w/ teacher if teacher provided
    latency_records = []

    # Create CSV log
    csv_path = os.path.join(save_path, "episode_log.csv")
    csv_file = open(csv_path, "w", newline=""); csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode","score","steps","mean_latency_ms","agreement_with_teacher"])

    while episodes_done < episodes:
        game.new_episode()
        spike_hist = deque(maxlen=window-1)
        for _ in range(window-1):
            spike_hist.append(np.zeros(len(action_channel_groups), dtype=np.float32))
        total_reward = 0.0; steps=0; rasters=[]; start_ts=time(); latencies=[]
        agreement_counts = 0; teacher_checks = 0

        # run ticks until episode finished
        for tick in neurons.loop(ticks_per_second=TICKS_PER_SECOND):
            t0 = time()
            state = game.get_state()

            # encode or simulate
            if allow_stim:
                encode_hybrid(neurons, state, plans, patterns, spatial_map)
                # capture spikes and build feature
                feat, counts = extract_spike_features(tick.analysis.spikes, spike_hist)
                spike_hist.append(counts)
            else:
                # simulate spikes correlated with teacher (if teacher exists) or weak baseline
                num_groups = len(action_channel_groups)
                if teacher is not None:
                    frame = preprocess(state.screen_buffer) if state else np.zeros((1,RESOLUTION[0],RESOLUTION[1]), dtype=np.float32)
                    t_act = teacher.get_action(frame)
                    base = np.zeros(num_groups, dtype=np.float32)
                    base[t_act % num_groups] = 5.0
                else:
                    base = np.ones(num_groups, dtype=np.float32) * 1.0
                counts = np.random.poisson(base).astype(np.float32)
                feat = np.concatenate(list(spike_hist) + [counts], axis=0)
                spike_hist.append(counts)

            rasters.append(counts)  # store per-tick counts (small)

            # prepare input for decoder
            xb = torch.from_numpy(feat).unsqueeze(0).float().to(device)
            with torch.no_grad():
                logits = spike_decoder(xb)
                act_idx = int(torch.argmax(logits, dim=1).cpu().numpy()[0])

            # apply action
            action_vec = [0]*game.get_available_buttons_size()
            if len(action_vec) >= 3: action_vec[act_idx]=1
            else: action_vec = [1 if i==act_idx else 0 for i in range(len(action_vec))]
            game.set_action(action_vec); game.advance_action(FRAME_REPEAT)

            # reward
            rew = game.get_last_reward()
            total_reward += rew; steps += 1

            # latency measurement
            lat_ms = (time() - t0) * 1000.0
            latencies.append(lat_ms)

            # teacher agreement check if teacher provided
            if teacher is not None:
                frame = preprocess(state.screen_buffer) if state else np.zeros((1,RESOLUTION[0],RESOLUTION[1]), dtype=np.float32)
                t_action = teacher.get_action(frame)
                teacher_checks += 1
                if t_action == act_idx: agreement_counts += 1

            # Optional online finetune: small supervised step towards teacher (if provided)
            if online_finetune and teacher is not None:
                model = spike_decoder; model.train()
                xb_ft = xb; yb_ft = torch.tensor([t_action], dtype=torch.long).to(device)
                # create small optimizer per-run (cheap) — OK for low-rate finetune
                opt = optim.Adam(model.parameters(), lr=1e-5)
                loss = nn.CrossEntropyLoss()(model(xb_ft), yb_ft)
                opt.zero_grad(); loss.backward(); opt.step()
                model.eval()

            if show_automap_flag:
                show_automap(game)

            if game.is_episode_finished():
                mean_lat = np.mean(latencies) if latencies else 0.0
                agreement = (agreement_counts/teacher_checks) if teacher_checks>0 else None
                episode_scores.append(total_reward)
                moving_avg = np.mean(episode_scores[-MOVING_AVG_WINDOW:])
                moving_avg_scores.append(moving_avg)
                action_agreements.append(agreement)
                latency_records.append(mean_lat)

                # save raster and per-episode logs
                rasters_np = np.stack(rasters, axis=0)  # (steps, action_groups)
                raster_path = os.path.join(save_path, f"raster_ep{episodes_done}.npy")
                np.save(raster_path, rasters_np)

                # Save a small png of spike counts over time
                plt.figure(figsize=(6,2))
                plt.imshow(rasters_np.T, aspect='auto', interpolation='nearest')
                plt.xlabel('tick'); plt.ylabel('action-group'); plt.title(f'Raster Ep{episodes_done}')
                pngpath = os.path.join(save_path, f"raster_ep{episodes_done}.png")
                plt.savefig(pngpath); plt.close()

                csv_writer.writerow([episodes_done, total_reward, steps, mean_lat, agreement if agreement is not None else "NA"])
                csv_file.flush()

                print(f"[RUN] Episode {episodes_done} score={total_reward:.2f} steps={steps} mean_latency_ms={mean_lat:.2f} agreement={agreement}")
                # feedback at episode end
                if allow_stim:
                    # graded magnitude: scale by percent of maximum possible (rough)
                    magnitude = min(1.0, max(0.0, total_reward/100.0))
                    plan = graded_feedback_plan(neurons, magnitude=magnitude, positive=(total_reward>0))
                    plan.run()
                # safety interrupt (only interrupt stim channels and feedback channels)
                neurons.interrupt(stim_channels); neurons.interrupt(feedback_channels_set)
                episodes_done += 1
                break

        # end episode loop

    csv_file.close()
    # Save summary plots
    plt.figure(); plt.plot(episode_scores, label='episode_score'); plt.plot(moving_avg_scores, label=f'moving_avg({MOVING_AVG_WINDOW})'); plt.legend(); plt.title('Scores'); plt.savefig(os.path.join(save_path,'scores.png')); plt.close()
    if any([a is not None for a in action_agreements]):
        plt.figure(); plt.plot([a if a is not None else 0 for a in action_agreements]); plt.title('Agreement with Teacher'); plt.savefig(os.path.join(save_path,'agreement.png')); plt.close()
    print("[RUN] Closed-loop finished. Logs saved to", save_path)

# -------------------------
# CLI + top-level orchestration
# -------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="Improved Hybrid DQN-teacher + Neuron-centric Spike Imitation")
    p.add_argument("--config", type=str, default=os.path.join(vzd.scenarios_path, "simpler_basic.cfg"))
    p.add_argument("--train-dqn", action="store_true")
    p.add_argument("--collect-spike-data", action="store_true")
    p.add_argument("--collect-samples", type=int, default=3000)
    p.add_argument("--train-spike-decoder", action="store_true")
    p.add_argument("--run-closed-loop", action="store_true")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--save-path", type=str, default=os.getcwd())
    p.add_argument("--allow-stim", action="store_true", help="Enable hardware stimulation (dangerous!). Must be set explicitly.")
    p.add_argument("--encoding", type=str, choices=["temporal","rate","spatial","hybrid"], default="hybrid")
    p.add_argument("--load-dqn", type=str, default=None)
    p.add_argument("--load-spike-decoder", type=str, default=None)
    p.add_argument("--online-finetune", action="store_true", help="Fine-tune spike decoder online during closed-loop using teacher")
    p.add_argument("--show-automap", action="store_true", help="Display ViZDoom top-down automap for debugging/interpretation")

    return p

def main():
    args = build_argparser().parse_args()

    # If no args, enable full pipeline with simulation (safe defaults)
    if len(sys.argv) == 1:
        print("No CLI arguments → running full default pipeline (collect → train → closed-loop)")
        args.collect_spike_data = True
        args.collect_samples = 5000
        args.train_spike_decoder = True
        args.run_closed_loop = True
        args.allow_stim = False  # software mode by default
        args.train_dqn = False  # only re-train teacher if you want

    set_seed()
    os.makedirs(args.save_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)
    if args.allow_stim:
        print("!!! WARNING: STIMULATION ENABLED. Ensure approvals and safety BEFORE running on real tissue !!!")

    show_automap_flag = bool(args.show_automap)

    # init Doom
    game = DoomGame(); game.load_config(args.config)

    # Make sure episodes don’t end instantly
    game.set_episode_timeout(2000)  # ~2000 tics per episode
    game.set_episode_start_time(10)  # skip the first frames to avoid spawn flicker

    game.set_window_visible(True)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()

    n_buttons = game.get_available_buttons_size()
    if n_buttons < 3:
        print("[WARN] env has <3 buttons - action mapping assumptions may need update")

    # ------------------------
    # Run episodes properly
    # ------------------------
    if args.run_closed_loop:
        print("[INFO] Running closed-loop control...")
        for ep in range(args.episodes):
            game.new_episode()
            while not game.is_episode_finished():
                state = game.get_state()
                if state is None:
                    continue

                # 👉 TODO: Replace with decoder prediction
                action_idx = 0
                action = [0] * n_buttons
                action[action_idx] = 1

                reward = game.make_action(action, FRAME_REPEAT)

            print(f"Episode {ep+1} finished, total reward {game.get_total_reward()}")

    # Build teacher (DQN)
    teacher = DQNAgent(action_size=3, device=device, load_model=args.load_dqn)
    if args.train_dqn:
        # simple training loop (fast version)
        print("[DQN] Training teacher (this can be long).")
        actions = []
        for a in range(3):
            vec = [0]*n_buttons
            vec[a if a<n_buttons else 0] = 1
            actions.append(vec)
        for epoch in range(TRAIN_EPOCHS):
            print(f"[DQN] epoch {epoch+1}/{TRAIN_EPOCHS}")
            game.new_episode()
            for step in trange(LEARNING_STEPS_PER_EPOCH, leave=False):
                state = preprocess(game.get_state().screen_buffer)
                a = teacher.get_action(state)
                r = game.make_action(actions[a], FRAME_REPEAT)
                done = game.is_episode_finished()
                next_state = preprocess(game.get_state().screen_buffer) if not done else np.zeros((1,RESOLUTION[0],RESOLUTION[1]), dtype=np.float32)
                teacher.store(state, a, r, next_state, done)
                teacher.train_step()
                if done: game.new_episode()
            teacher.sync_target()
        torch.save(teacher.q_net.state_dict(), os.path.join(args.save_path, "dqn_teacher.pth"))
        print("[DQN] Teacher training complete.")

    # open cl
    with cl.open() as neurons:
        # precompile plans (if allowed)
        plans = build_precompiled_plans(neurons) if args.allow_stim else {}
        # prepare patterns for rate coding (ChannelSets with incremental counts)
        patterns = None
        if args.allow_stim:
            # prepare incremental ChannelSets from stim_channels
            stim_list = list(stim_channels.channels) if hasattr(stim_channels, 'channels') else list(stim_channels)
            patterns = []
            for k in range(1, len(stim_list)+1):
                patterns.append(neurons.create_stim_plan())
                patterns[-1].interrupt(cl.ChannelSet(*stim_list[:k]))
                patterns[-1].stim(cl.ChannelSet(*stim_list[:k]), stim_design, cl.BurstDesign(100, 10))
        # spatial map example (split 4 sectors)
        spatial_map = {}
        if args.allow_stim:
            sectors = 4
            for s in range(sectors):
                p = neurons.create_stim_plan(); p.interrupt(stim_channels)
                # rotate selection of electrodes per sector for demonstration
                ch = cl.ChannelSet(stim_list[s % len(stim_list)])
                p.stim(ch, stim_design, cl.BurstDesign(150, 20+(s*10)))
                spatial_map[s] = p

        # Collect spike dataset
        if args.collect_spike_data:
            X,y = collect_spike_dataset(game, neurons, teacher, target_samples=args.collect_samples,
                                        max_episodes=50, plans=plans, patterns=patterns, spatial_map=spatial_map,
                                        allow_stim=args.allow_stim, window=SPIKE_WINDOW, save_path=args.save_path,
                                        show_automap_flag=show_automap_flag)
            print("[DATA] Collected:", X.shape, y.shape)

        # Train spike decoder
        spike_decoder = None
        sd_path = os.path.join(args.save_path, "spike_decoder.pth")
        if args.train_spike_decoder:
            Xpath = os.path.join(args.save_path, "spike_X.npy"); ypath = os.path.join(args.save_path, "spike_y.npy")
            if os.path.exists(Xpath) and os.path.exists(ypath):
                X = np.load(Xpath); y = np.load(ypath)
            else:
                print("[SPK] dataset not found; collecting now...")
                X,y = collect_spike_dataset(game, neurons, teacher, target_samples=args.collect_samples,
                                            max_episodes=50, plans=plans, patterns=patterns, spatial_map=spatial_map,
                                            allow_stim=args.allow_stim, window=SPIKE_WINDOW, save_path=args.save_path,
                                            show_automap_flag=show_automap_flag)
            if len(X) < SPIKE_DATA_MIN_SAMPLES:
                print(f"[SPK] Warning: small dataset ({len(X)} samples). Consider collecting more.")
            # Simple normalization for stability
            mu = X.mean(axis=0, keepdims=True)
            std = X.std(axis=0, keepdims=True) + 1e-6
            X = (X - mu) / std
            np.save(os.path.join(args.save_path, "spike_X_norm.npy"), X)
            spike_decoder = train_spike_decoder(X,y,device=device, epochs=SPIKE_DECODER_EPOCHS, save_path=args.save_path)

        # Optionally load spike decoder
        if args.load_spike_decoder:
            ld = args.load_spike_decoder
            if os.path.exists(ld):
                spike_decoder = SpikeDecoder(input_dim=len(action_channel_groups)*SPIKE_WINDOW, hidden=SPIKE_DECODER_HID, out=3)
                spike_decoder.load_state_dict(torch.load(ld, map_location=device)); spike_decoder.to(device)
                print("[SPK] Loaded spike decoder", ld)
            else:
                print("[SPK] load path not found:", ld)

        # Run closed-loop
        if args.run_closed_loop:
            if spike_decoder is None:
                # try default path
                if os.path.exists(sd_path):
                    spike_decoder = SpikeDecoder(input_dim=len(action_channel_groups)*SPIKE_WINDOW, hidden=SPIKE_DECODER_HID, out=3)
                    spike_decoder.load_state_dict(torch.load(sd_path, map_location=device)); spike_decoder.to(device)
                    print("[SPK] Loaded spike decoder from default", sd_path)
                else:
                    raise RuntimeError("No spike decoder available. Train or provide one.")
            run_closed_loop_neuron_control(game, neurons, spike_decoder, plans=plans, patterns=patterns,
                                           spatial_map=spatial_map, episodes=args.episodes, window=SPIKE_WINDOW,
                                           allow_stim=args.allow_stim, save_path=args.save_path, device=device,
                                           online_finetune=args.online_finetune, teacher=teacher,
                                           show_automap_flag=show_automap_flag)

    game.close()
    print("[INFO] All done. Logs saved to:", args.save_path)

if __name__ == "__main__":
    main()
