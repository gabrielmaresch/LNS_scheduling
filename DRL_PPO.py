from __future__ import annotations
import random
import math
from dataclasses import dataclass, field
from typing import Dict, Callable
from pathlib import Path
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rws import RWS, rws_lns
from rws_instance_loader import load_instance_and_schedule
import matplotlib.pyplot as plt

# ============================================================
# ACTOR-CRITIC NETWORK (PPO)
# ============================================================
class ActorCritic(nn.Module):
    """Actor-Critic network for discrete operator + discrete severity/temperature.

    Actions:
    - destroy, repair: categorical
    - severity: categorical with 10 bins (10%-100%)
    - temperature: categorical with 50 bins (0.1 - 5.0)
    """
    def __init__(self, state_dim, n_destroy, n_repair, n_severity=10, n_temp=50):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        # Discrete action heads (categorical)
        self.destroy_head = nn.Linear(64, n_destroy)
        self.repair_head = nn.Linear(64, n_repair)
        # Discrete heads for severity and temperature
        self.severity_head = nn.Linear(64, n_severity)
        self.temp_head = nn.Linear(64, n_temp)

        # Value function
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        features = self.shared(x)
        return (
            self.destroy_head(features),      # logits for destroy
            self.repair_head(features),       # logits for repair
            self.severity_head(features),     # logits for severity bins
            self.temp_head(features),         # logits for temperature bins
            self.value_head(features)         # scalar value
        )

# ============================================================
# METRICS MONITOR
# ============================================================
class MetricsMonitor:
    """Tracks and saves training metrics for analysis."""
    def __init__(self, save_path: Path = None):
        self.save_path = save_path
        self.metrics = {}
    
    def record(self, epoch: int, **kwargs):
        """Record metrics for an epoch."""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def save_csv(self, filename: str = "training_metrics.csv"):
        """Save metrics to CSV file."""
        import csv
        if not self.metrics:
            print("No metrics to save")
            return
        
        save_path = self.save_path / filename if self.save_path else Path(filename)
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            keys = list(self.metrics.keys())
            writer.writerow(keys)
            num_rows = len(self.metrics[keys[0]])
            for i in range(num_rows):
                row = [self.metrics[k][i] if i < len(self.metrics[k]) else "" for k in keys]
                writer.writerow(row)
        print(f"✓ Metrics saved to {save_path}")
    
    def get_summary(self):
        """Get summary statistics."""
        if not self.metrics:
            return {}
        summary = {}
        for key, values in self.metrics.items():
            if isinstance(values[0], (int, float)):
                summary[key] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "final": values[-1]
                }
        return summary

# ============================================================
# DRL-ALNS WITH PPO
# ============================================================
@dataclass
class drl_alns:
    instance: RWS.Instance
    schedule: RWS.Schedule
    destroy_operators: Dict[str, Callable]
    repair_operators: Dict[str, Callable]
    gamma: float = 0.99
    lr: float = 3e-4
    clip_eps: float = 0.2
    update_epochs: int = 5
    rollout_length: int = 10
    exploration_after_stagnation: int = 20  # Force exploration after N stagnation rounds
    tabu_length: int = 8  # Tabu list size
    # PPO / training hyperparameters
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.1
    max_grad_norm: float = 0.5

    def __post_init__(self):
        self.lns = rws_lns(
            instance=self.instance,
            incumbent=self.schedule
        )
        self.destroy_names = list(self.destroy_operators.keys())
        self.repair_names = list(self.repair_operators.keys())
        # State vector: 6 features (see Table 1, removed search-budget)
        self.state_dim = 6
        # ActorCritic now expects n_severity=10, n_temp=50 by default
        self.model = ActorCritic(
            self.state_dim,
            len(self.destroy_names),
            len(self.repair_names),
            n_severity=10,
            n_temp=50,
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.best_conflicts = self._count_conflicts(self.schedule)
        self.current_conflicts = self.best_conflicts
        self.prev_improved = 0
        self.prev_best = 0
        self.prev_accepted = 0
        self.stagnation = 0
        self.iteration = 0
        self.metrics_monitor = MetricsMonitor(save_path=Path(__file__).parent)
        
        # === Tabu list: prevent cycling to same destroyed patterns ===
        # (This is a constraint on action space, not operator weighting)
        self.tabu_history = deque(maxlen=self.tabu_length)
        self.tabu_signatures = {}  # signature -> count

    # --------------------------------------------------------
    def _count_conflicts(self, schedule):
        return int(sum(schedule.count_total_violations().values()))

    # --------------------------------------------------------
    def _get_state(self):
        """Return 6-element state vector per Table 1 (search-budget removed):
        [best_improved, current_accepted, current_improved, is_current_best,
         cost_diff_best, stagnation_count]
        """
        # best_improved: whether best improved in last step
        best_improved = float(self.prev_best)
        # current_accepted: whether last candidate was accepted
        current_accepted = float(self.prev_accepted)
        # current_improved: whether last accepted move improved incumbent
        current_improved = float(self.prev_improved)
        # is_current_best: whether current equals best
        is_current_best = 1.0 if self.current_conflicts <= self.best_conflicts else 0.0
        # cost difference: -1.0 when current <= best, else normalized difference
        if self.current_conflicts <= self.best_conflicts:
            cost_diff_best = -1.0
        else:
            cost_diff_best = (self.current_conflicts - self.best_conflicts) / max(1, self.best_conflicts)
        stagnation_count = float(self.stagnation)

        return np.array([
            best_improved,
            current_accepted,
            current_improved,
            is_current_best,
            cost_diff_best,
            stagnation_count,
        ], dtype=np.float32)

    # ========== NEW: TABU & EXPLORATION METHODS ==========
    def _destroyed_signature(self, destroyed_pairs: list[tuple[int, int]]) -> tuple:
        """Extract worker/day sets from destroyed pairs."""
        workers = frozenset(w for w, _ in destroyed_pairs)
        days = frozenset(d for _, d in destroyed_pairs)
        return (workers, days)
    
    def _is_tabu(self, destroyed_pairs: list[tuple[int, int]]) -> bool:
        """Check if this destroy signature is in tabu list."""
        if not destroyed_pairs:
            return False
        sig = self._destroyed_signature(destroyed_pairs)
        return sig in self.tabu_signatures
    
    def _record_tabu(self, destroyed_pairs: list[tuple[int, int]]) -> None:
        """Add destroy signature to tabu history."""
        if not destroyed_pairs:
            return
        sig = self._destroyed_signature(destroyed_pairs)
        self.tabu_history.append(sig)
        self.tabu_signatures[sig] = self.tabu_signatures.get(sig, 0) + 1
        # When deque evicts oldest (maxlen), manually clean it up
        if len(self.tabu_history) == self.tabu_length:
            # deque will auto-evict, but we need to track it
            pass
    
    def _update_tabu_after_eviction(self):
        """Clean up tabu_signatures for entries no longer in history."""
        current_sigs = set(self.tabu_history)
        to_remove = [sig for sig in self.tabu_signatures if sig not in current_sigs]
        for sig in to_remove:
            del self.tabu_signatures[sig]

    # =========================================
    # =========================================
    def _select_action(self, state):
        """Select discrete actions: destroy_idx, repair_idx, severity_bin, temp_bin.

        Severity bins: 10 values mapping to [0.1, 1.0]
        Temperature bins: 50 values mapping to [0.1, 5.0]
        Returns integer indices and the summed log-prob for PPO.
        """
        state_t = torch.tensor(state).float().unsqueeze(0)
        logits_d, logits_r, logits_sev, logits_temp, value = self.model(state_t)

        # Categorical distributions for all discrete actions
        dist_d = torch.distributions.Categorical(logits=logits_d)
        dist_r = torch.distributions.Categorical(logits=logits_r)
        dist_sev = torch.distributions.Categorical(logits=logits_sev)
        dist_temp = torch.distributions.Categorical(logits=logits_temp)

        # Forced exploration: random indices for operators and bins
        if self.stagnation >= self.exploration_after_stagnation:
            self.use_exploration_phase = True
            a_d = random.randint(0, len(self.destroy_names) - 1)
            a_r = random.randint(0, len(self.repair_names) - 1)
            a_sev = random.randint(0, logits_sev.shape[-1] - 1)
            a_temp = random.randint(0, logits_temp.shape[-1] - 1)
            log_prob_d = torch.log_softmax(logits_d, dim=-1)[:, a_d].squeeze()
            log_prob_r = torch.log_softmax(logits_r, dim=-1)[:, a_r].squeeze()
            log_prob_sev = torch.log_softmax(logits_sev, dim=-1)[:, a_sev].squeeze()
            log_prob_temp = torch.log_softmax(logits_temp, dim=-1)[:, a_temp].squeeze()
            total_log_prob = log_prob_d + log_prob_r + log_prob_sev + log_prob_temp
        else:
            self.use_exploration_phase = False
            a_d = dist_d.sample().item()
            a_r = dist_r.sample().item()
            a_sev = dist_sev.sample().item()
            a_temp = dist_temp.sample().item()
            log_prob_d = dist_d.log_prob(torch.tensor(a_d))
            log_prob_r = dist_r.log_prob(torch.tensor(a_r))
            log_prob_sev = dist_sev.log_prob(torch.tensor(a_sev))
            log_prob_temp = dist_temp.log_prob(torch.tensor(a_temp))
            total_log_prob = log_prob_d + log_prob_r + log_prob_sev + log_prob_temp

        # Map bins to actionable floats
        severity = (float(a_sev) + 1.0) / 10.0  # 1->10% ... 10->100%
        temperature = 0.1 + (float(a_temp) / 49.0) * (5.0 - 0.1)

        return int(a_d), int(a_r), int(a_sev), int(a_temp), total_log_prob.squeeze(), value.squeeze()

    # --------------------------------------------------------
    # STEP FUNCTION: Execute destroy-repair cycle, compute reward
    # --------------------------------------------------------
    def step(self, a_d, a_r, a_sev_bin, a_temp_bin, cumulative_reward=0.0):
        destroy_name = self.destroy_names[a_d]
        repair_name = self.repair_names[a_r]
        destroy_op = self.destroy_operators[destroy_name]
        repair_op = self.repair_operators[repair_name]
        
        # Initialize fixed vars for fresh LNS step
        self.lns.incumbent = self.schedule
        self.lns.contender = None
        self.lns._initialize_fixed_vars()  # ← CRITICAL: populate fixed_vars from incumbent
        self.lns.destroyed_pairs = []
        
        # Convert bins to actual values
        severity = (float(a_sev_bin) + 1.0) / 10.0
        temperature = 0.1 + (float(a_temp_bin) / 49.0) * (5.0 - 0.1)

        # Execute destroy with severity controlling fraction
        try:
            destroyed = destroy_op(self.lns, severity)
            self.lns.destroyed_pairs = destroyed
        except Exception as e:
            print(f"  ✗ Destroy failed: {e}")
            return self._get_state(), -1.0
        
        # Execute repair
        try:
            repair_op(self.lns)
            contender = self.lns.contender
        except Exception as e:
            print(f"  ✗ Repair failed: {e}")
            contender = None
        
        if contender is None:
            # record that last candidate was not accepted
            self.prev_accepted = 0
            return self._get_state(), -1.0
        
        # Evaluate move
        new_conf = self._count_conflicts(contender)
        old_conf = self.current_conflicts
        delta = new_conf - old_conf
        
        # Simulated annealing acceptance
        accepted = False
        acceptance_prob = 0.0
        if delta <= 0:
            accepted = True
            acceptance_prob = 1.0
        else:
            acceptance_prob = math.exp(-delta / max(temperature, 0.1))
            if random.random() < acceptance_prob:
                accepted = True
        
        # Compute reward
        reward = 0.0
        improved = 0
        new_best = 0
        
        if accepted:
            self.schedule = contender
            self.current_conflicts = new_conf

            # Primary reward: relative improvement (normalized by baseline)
            baseline_penalty = old_conf / max(1, 50)  # Normalize by typical conflict count
            reward = -(new_conf - old_conf) / max(1, baseline_penalty)  # Reward improvement

            # Distinguish between (a) improvement but not a new global best (small bonus)
            # and (b) new global best (larger bonus).
            if new_conf < old_conf:
                improved = 1
                if new_conf < self.best_conflicts:
                    # New global best
                    self.best_conflicts = new_conf
                    reward += 10.0
                    new_best = 1
                else:
                    # Small bonus for improving current solution (but not global best)
                    reward += 0.5
                self.stagnation = 0
            else:
                self.stagnation = 0  # Reset stagnation on any accepted move
            
            # Record tabu after acceptance
            self._record_tabu(self.lns.destroyed_pairs)
            self.prev_accepted = 1
        else:
            # DYNAMIC rejection penalty: worse rejections get higher penalty
            # Delta = new_conf - old_conf (positive = worse)
            #penalty_magnitude = (delta / max(1, old_conf)) * 3.0  # Scale by severity of rejection
            #reward = -min(penalty_magnitude, 2.0)  # Cap at -2.0 (avoid extreme penalties)
            
            self.stagnation += 1
            self.prev_accepted = 0
        
        # Periodically clean tabu list
        if self.iteration % self.tabu_length == 0:
            self._update_tabu_after_eviction()
        
        self.prev_improved = improved
        self.prev_best = new_best
        self.iteration += 1
        
        cumulative_reward += reward
        
        # ====== DETAILED STEP METRICS ======
        status = "[ACCEPT]" if accepted else "[REJECT]"
        improvement_marker = "DOWN" if new_conf < old_conf else "SAME" if new_conf == old_conf else "UP"
        best_marker = "NEW BEST!" if new_best else ""
        acceptance_prob_pct = (acceptance_prob * 100) if delta > 0 else 100.0
        exploration_marker = " (EXPLORATION)" if self.use_exploration_phase else ""
        
        print(
            f"  Step {self.iteration:4d} | "
            f"D:{destroy_name:20s} R:{repair_name:18s} | "
            f"Sev:{severity:.3f} Tmp:{temperature:.2f} | "
            f"Conflicts:{old_conf:3d}->{new_conf:3d}({improvement_marker:4s}) Best:{self.best_conflicts:3d} | "
            f"AcceptProb:{acceptance_prob_pct:5.1f}% Reward:{reward:+.3f} | "
            f"{status} {best_marker}{exploration_marker} | "
            f"Stagnation:{self.stagnation}"
        )
        
        return self._get_state(), reward

    # --------------------------------------------------------
    def _ppo_update(self, states, actions_discrete, old_log_probs, returns, advantages):
        """PPO update for discrete action space (destroy, repair, severity_bin, temp_bin).

        Args:
            states: [batch_size, state_dim] tensor
            actions_discrete: [batch_size, 4] integer indices for (destroy, repair, severity_bin, temp_bin)
            old_log_probs: [batch_size] old probabilities
            returns: [batch_size] discounted returns
            advantages: [batch_size] advantage estimates
        """
        last_policy = 0
        last_value = 0
        last_entropy = 0
        
        for _ in range(self.update_epochs):
            logits_d, logits_r, logits_sev, logits_temp, values = self.model(states)

            # Categorical distributions for all discrete actions
            dist_d = torch.distributions.Categorical(logits=logits_d)
            dist_r = torch.distributions.Categorical(logits=logits_r)
            dist_sev = torch.distributions.Categorical(logits=logits_sev)
            dist_temp = torch.distributions.Categorical(logits=logits_temp)

            # New log-probs for the actions taken (sum of four categorical log-probs)
            new_log_probs = (
                dist_d.log_prob(actions_discrete[:, 0]) +
                dist_r.log_prob(actions_discrete[:, 1]) +
                dist_sev.log_prob(actions_discrete[:, 2]) +
                dist_temp.log_prob(actions_discrete[:, 3])
            )

            # Entropy bonus from all discrete distributions
            entropy = (
                dist_d.entropy().mean()
                + dist_r.entropy().mean()
                + dist_sev.entropy().mean()
                + dist_temp.entropy().mean()
            )

            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
            
            # Policy loss: standard PPO on combined log-probs
            policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
            
            # Value loss: MSE between predicted and target returns (with lower weight)
            value_loss = nn.functional.mse_loss(values.squeeze(), returns)
            
            # Combined loss using configured coefficients
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()
            
            last_policy = policy_loss.item()
            last_value = value_loss.item()
            last_entropy = entropy.item()
        
        return last_policy, last_value, last_entropy

    # --------------------------------------------------------
    def train(self, iterations=2000):
        state = self._get_state()
        # Logger for plotting
        log = {
            "destroy": [],
            "repair": [],
            "severity": [],
            "temperature": [],
            "reward": [],
            "current_conflicts": [],
            "best_conflicts": [],
            "stagnation": [],
            "episode_return": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "accepted_count": [],
            "cumulative_rewards": [],
        }
        
        print("\n" + "="*150)
        print(f"{'Training DRL-ALNS with PPO':^150}")
        print("="*150 + "\n")
        
        for it in range(iterations):
            states_list = []
            actions_discrete_list = []
            actions_continuous_list = []
            rewards = []
            log_probs = []
            values = []
            cumulative_reward = 0.0
            accepted_count = 0
            
            print(f"\n{'-'*150}")
            print(f"EPOCH {it+1:3d}/{iterations} | Instance: {self.instance.num_workers} workers, {self.instance.num_days} days")
            print(f"{'-'*150}\n")
            
            for step_i in range(self.rollout_length):
                a_d, a_r, a_sev, a_temp, log_p, value = self._select_action(state)
                next_state, reward = self.step(a_d, a_r, a_sev, a_temp, cumulative_reward)

                # Map bins to actionable floats for logging
                severity = (float(a_sev) + 1.0) / 10.0
                temperature = 0.1 + (float(a_temp) / 49.0) * (5.0 - 0.1)

                if reward > -1.0:  # Count accepted moves
                    accepted_count += 1
                cumulative_reward += reward

                # Log metrics
                log["destroy"].append(self.destroy_names[a_d])
                log["repair"].append(self.repair_names[a_r])
                log["severity"].append(severity)
                log["temperature"].append(temperature)
                log["reward"].append(reward)
                log["current_conflicts"].append(self.current_conflicts)
                log["best_conflicts"].append(self.best_conflicts)
                log["stagnation"].append(self.stagnation)

                # Store for PPO update (destroy, repair, severity_bin, temp_bin)
                states_list.append(state)
                actions_discrete_list.append([a_d, a_r, a_sev, a_temp])
                rewards.append(reward)
                log_probs.append(log_p)
                values.append(value)

                state = next_state
            
            episode_return = sum(rewards)
            acceptance_rate = accepted_count / self.rollout_length * 100
            log["episode_return"].append(episode_return)
            log["accepted_count"].append(accepted_count)
            log["cumulative_rewards"].append(cumulative_reward)
            
            # ====== EPOCH SUMMARY ======
            print(f"\n{'-'*150}")
            print(f"EPOCH {it+1:3d} SUMMARY")
            print(f"{'-'*150}")
            
            exploration_status = "🔍 EXPLORATION PHASE" if self.use_exploration_phase else "🎯 NORMAL"
            print(
                f"  Best Conflicts:      {self.best_conflicts:3d}\n"
                f"  Current Conflicts:   {self.current_conflicts:3d}\n"
                f"  Stagnation Counter:  {self.stagnation:3d}  ({exploration_status})\n"
                f"  Episode Return:      {episode_return:+.3f}\n"
                f"  Cumulative Reward:   {cumulative_reward:+.3f}\n"
                f"  Acceptance Rate:     {acceptance_rate:.1f}% ({accepted_count}/{self.rollout_length})\n"
            )
            
            print()
            
            # ---- Prepare data for PPO update ----
            states = torch.from_numpy(np.array(states_list, dtype=np.float32))
            actions_discrete = torch.tensor(actions_discrete_list, dtype=torch.long)
            old_log_probs = torch.stack(log_probs).detach()
            values = torch.stack(values)
            
            # ===== Compute GAE advantages and returns =====
            # values: tensor of shape [T]
            values_t = values.detach().squeeze()
            T = len(rewards)

            # compute last value for bootstrap
            with torch.no_grad():
                _, _, _, _, last_v = self.model(torch.from_numpy(state).float().unsqueeze(0))
                last_value = last_v.squeeze().item()

            rewards_t = torch.tensor(rewards, dtype=torch.float32)
            advantages = torch.zeros(T, dtype=torch.float32)
            gae = 0.0
            for t in reversed(range(T)):
                next_value = last_value if t == T - 1 else values_t[t + 1].item()
                delta = rewards_t[t].item() + self.gamma * next_value - values_t[t].item()
                gae = delta + self.gamma * self.gae_lambda * gae
                advantages[t] = gae

            returns = advantages + values_t

            # Normalize advantages and returns for stability
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            returns_normalized = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # PPO update with normalized returns
            pl, vl, ent = self._ppo_update(
                states,
                actions_discrete,
                old_log_probs,
                returns_normalized,
                advantages
            )
            log["policy_loss"].append(pl)
            log["value_loss"].append(vl)
            log["entropy"].append(ent)
            
            print(
                f"  Policy Loss:         {pl:.6f}\n"
                f"  Value Loss:          {vl:.6f}\n"
                f"  Entropy:             {ent:.6f}\n"
            )
            print(f"{'-'*150}\n")
            
            # Record metrics
            self.metrics_monitor.record(
                epoch=it + 1,
                best_conflicts=self.best_conflicts,
                current_conflicts=self.current_conflicts,
                episode_return=episode_return,
                cumulative_reward=cumulative_reward,
                acceptance_rate=acceptance_rate,
                policy_loss=pl,
                value_loss=vl,
                entropy=ent,
                stagnation=self.stagnation
            )
            
            # Early stopping conditions
            if self.best_conflicts == 0:
                print(f"\n{'*'*20} SOLVED with zero conflicts at iteration {it+1}! {'*'*20}\n")
                break
            if self.stagnation >= 100:
                print(f"\n{'!'*20} Stopping: stagnation >= 100 at iteration {it+1} {'!'*20}\n")
                break
        
        print("\n" + "="*150)
        print(f"Training completed. Best solution: {self.best_conflicts} conflicts")
        print("="*150 + "\n")
        
        # Save metrics
        self.metrics_monitor.save_csv("drl_alns_training_metrics.csv")
        
        return self.schedule, log

# =========================
# DESTROY / REPAIR HELPERS
# =========================
def _make_repair_operator(model_path: Path, solver_name: str, timeout_seconds: int):
    def op(lns: rws_lns):
        return lns.repair_exact(
            model_path=model_path,
            solver_name=solver_name,
            timeout_seconds=timeout_seconds
        )
    return op

def _ceil_fraction_count(total: int, fraction: float) -> int:
    """Compute at least one unit from a fraction of total"""
    return max(1, int(total * fraction))

def _take_ranked_ids(ranked_dict, k: int):
    """Return top-k IDs based on a ranking dictionary"""
    sorted_items = sorted(ranked_dict.items(), key=lambda x: x[1], reverse=True)
    return [entity_id for entity_id, _ in sorted_items[:k]]

# -------------------------
# Destroy Operators (use severity)
# -------------------------
def _make_destroy_random_workers():
    def op(lns: rws_lns, severity: float):
        k = _ceil_fraction_count(lns.instance.num_workers, severity)
        ids = random.sample(range(lns.instance.num_workers), k)
        return lns.destroy_worker(ids)
    return op

def _make_destroy_random_days():
    def op(lns: rws_lns, severity: float):
        k = _ceil_fraction_count(lns.instance.num_days, severity)
        ids = random.sample(range(lns.instance.num_days), k)
        return lns.destroy_day(ids)
    return op

def _make_destroy_worst_workers():
    def op(lns: rws_lns, severity: float):
        schedule = lns.incumbent
        k = _ceil_fraction_count(lns.instance.num_workers, severity)
        ids = _take_ranked_ids(schedule.worker_ranked_by_violations, k)
        return lns.destroy_worker(ids)
    return op

def _make_destroy_worst_days():
    def op(lns: rws_lns, severity: float):
        schedule = lns.incumbent
        k = _ceil_fraction_count(lns.instance.num_days, severity)
        ids = _take_ranked_ids(schedule.days_ranked_by_violations, k)
        return lns.destroy_day(ids)
    return op

# def _make_destroy_worst_worker_local_days(window_size: int = 5):
#     def op(lns: rws_lns, severity: float = None):
#         """Destroy local window around worst worker. Severity unused (uses fixed window)."""
#         schedule = lns.incumbent
#         worker = max(
#             schedule.worker_ranked_by_violations,
#             key=schedule.worker_ranked_by_violations.get,
#             default=0
#         )
#         max_start = max(0, lns.instance.num_days - window_size)
#         start = random.randint(0, max_start)
#         days = list(range(start, start + window_size))
#         freed = []
#         for d in days:
#             if d < len(schedule.assignment) and worker < len(schedule.assignment[d]):
#                 key = (d, worker)
#                 if key in lns.fixed_vars:
#                     del lns.fixed_vars[key]
#                 freed.append(key)
#         return freed
#     return op

# def _make_destroy_worst_day_top_workers(num_workers: int = 3):
#     def op(lns: rws_lns, severity: float = None):
#         """Destroy top-k workers on worst day. Severity unused (uses fixed k)."""
#         schedule = lns.incumbent
#         day = max(
#             schedule.days_ranked_by_violations,
#             key=schedule.days_ranked_by_violations.get,
#             default=0
#         )
#         ranked = sorted(
#             schedule.worker_ranked_by_violations.items(),
#             key=lambda x: x[1],
#             reverse=True
#         )
#         workers = [w for w, _ in ranked[:num_workers]]
#         freed = []
#         for w in workers:
#             key = (day, w)
#             if key in lns.fixed_vars:
#                 del lns.fixed_vars[key]
#             freed.append(key)
#         return freed
#     return op

def _smooth(x, k=30):
    if len(x) < k:
        return x
    import numpy as np
    return np.convolve(x, np.ones(k)/k, mode="valid")

def plot_training(log):
    #Plot training metrics: episode return, conflicts, losses, entropy, acceptance rate.
    import matplotlib.pyplot as plt
    plt.figure(figsize=(18, 12))
    
    # -----------------------
    # 1) Episode return
    # -----------------------
    plt.subplot(2, 3, 1)
    if "episode_return" in log and log["episode_return"]:
        smoothed = _smooth(log["episode_return"], 5)
        plt.plot(smoothed, linewidth=2, label="Smoothed (k=5)")
        plt.plot(log["episode_return"], alpha=0.3, label="Raw")
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.title("Episode Return Over Time", fontsize=12, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # -----------------------
    # 2) Best & Current Conflicts
    # -----------------------
    plt.subplot(2, 3, 2)
    if "best_conflicts" in log and log["best_conflicts"]:
        epochs = list(range(len(log["best_conflicts"])))
        plt.plot(epochs, log["best_conflicts"], marker='o', label="Best", linewidth=2)
    if "current_conflicts" in log and log["current_conflicts"]:
        plt.plot(epochs, log["current_conflicts"], marker='s', label="Current", linewidth=2, alpha=0.7)
    plt.title("Conflicts Over Time", fontsize=12, fontweight='bold')
    plt.xlabel("Step")
    plt.ylabel("Number of Conflicts")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # -----------------------
    # 3) PPO Losses
    # -----------------------
    plt.subplot(2, 3, 3)
    if "policy_loss" in log and log["policy_loss"]:
        smoothed_pl = _smooth(log["policy_loss"], 5)
        plt.plot(smoothed_pl, linewidth=2, label="Policy Loss (smoothed)")
    if "value_loss" in log and log["value_loss"]:
        smoothed_vl = _smooth(log["value_loss"], 5)
        plt.plot(smoothed_vl, linewidth=2, label="Value Loss (smoothed)")
    plt.title("Loss Curves", fontsize=12, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # -----------------------
    # 4) Policy Entropy
    # -----------------------
    plt.subplot(2, 3, 4)
    if "entropy" in log and log["entropy"]:
        smoothed_ent = _smooth(log["entropy"], 5)
        plt.plot(smoothed_ent, linewidth=2, label="Entropy (smoothed)")
        plt.plot(log["entropy"], alpha=0.3, label="Raw")
    plt.title("Policy Entropy (Exploration)", fontsize=12, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Entropy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # -----------------------
    # 5) Acceptance Rate
    # -----------------------
    plt.subplot(2, 3, 5)
    if "accepted_count" in log and log["accepted_count"]:
        acceptance_rates = [count / 32 * 100 for count in log["accepted_count"]]  # Assuming rollout_length=32
        smoothed_ar = _smooth(acceptance_rates, 5)
        plt.plot(smoothed_ar, linewidth=2, label="Acceptance Rate (smoothed)")
        plt.plot(acceptance_rates, alpha=0.3, label="Raw")
        plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, label="Target ~50%")
    plt.title("Move Acceptance Rate", fontsize=12, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Acceptance Rate (%)")
    plt.ylim([0, 100])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # -----------------------
    # 6) Stagnation Counter
    # -----------------------
    plt.subplot(2, 3, 6)
    if "stagnation" in log and log["stagnation"]:
        plt.plot(log["stagnation"], marker='x', linewidth=2, color='orange')
        plt.axhline(y=100, color='r', linestyle='--', alpha=0.5, label="Early stop threshold")
    plt.title("Stagnation Counter", fontsize=12, fontweight='bold')
    plt.xlabel("Step")
    plt.ylabel("Steps Since Improvement")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    instance_path = base / "Instances1-50" / "Example1.txt"
    instance, schedule = load_instance_and_schedule(
        file_path=instance_path,
        cyclicity=True
    )
    print(f"\n=== Solving instance: {instance_path.name} ===")
    model_path = base / "rws_instance.mzn"
    repair_ops = {
        "repair_chuffed_fast": _make_repair_operator(model_path, "chuffed", 3),
        "repair_gecode_fast": _make_repair_operator(model_path, "gecode", 3),
        "repair_chuffed_long": _make_repair_operator(model_path, "chuffed", 15),
        "repair_gecode_long": _make_repair_operator(model_path, "gecode", 15),
    }
    destroy_ops = {
        "destroy_worst_workers": _make_destroy_worst_workers(),
        "destroy_random_workers": _make_destroy_random_workers(),
        "destroy_worst_days": _make_destroy_worst_days(),
        "destroy_random_days": _make_destroy_random_days(),
        #"destroy_worst_worker_window5": _make_destroy_worst_worker_local_days(5),
        #"destroy_worst_day_top3_workers": _make_destroy_worst_day_top_workers(3),
    }
    solver = drl_alns(
        instance=instance,
        schedule=schedule,
        destroy_operators=destroy_ops,
        repair_operators=repair_ops,
    )
    final_schedule, log = solver.train(iterations=2000)
    plot_training(log)
    print("\nFinal schedule:")
    final_schedule.display_schedule()
    final_schedule.display_violations()