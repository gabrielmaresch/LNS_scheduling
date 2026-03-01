from __future__ import annotations
import random
import math
import json
import re
from dataclasses import dataclass
from typing import Dict, Callable
from pathlib import Path
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from time import perf_counter
from rws import RWS, rws_lns
from rws_instance_loader import load_instance_and_schedule
from multiarm_bandit import (
    _make_destroy_forbidden_sequences as _mab_make_destroy_forbidden_sequences,
    _make_destroy_max_border_operators as _mab_make_destroy_max_border_operators,
    _make_destroy_streak as _mab_make_destroy_streak,
    _make_destroy_streak_around_worst_worker as _mab_make_destroy_streak_around_worst_worker,
    _make_destroy_random_days as _mab_make_destroy_random_days,
    _make_destroy_random_window as _mab_make_destroy_random_window,
    _make_destroy_random_workers as _mab_make_destroy_random_workers,
    _make_destroy_worst_days as _mab_make_destroy_worst_days,
    _make_destroy_worst_workers as _mab_make_destroy_worst_workers,
    _make_repair_operator as _mab_make_repair_operator,
    _make_repair_tabu_operator as _mab_make_repair_tabu_operator,
)
import matplotlib.pyplot as plt

"""
Think about reward function. 
Maybe we shouldn't train temperature and just keep it as in bandit
Minizinc time?
Code some severity suggestions? (eg, start low, high if high stagnation etc)
"""


# ============================================================
# ACTOR-CRITIC NETWORK (PPO)
# ============================================================
class ActorCritic(nn.Module):
    """Actor-Critic network for discrete operator + discrete severity/temperature.

    Actions:
    - destroy, repair: categorical
    - severity: categorical with 10 bins (10%-100%) --> as in the paper
    - temperature: categorical with 50 bins (0.1 - 5.0) --> as in the paper
    """
    def __init__(self, state_dim, n_destroy, n_repair, n_severity=10, n_temp=50):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64), # --> as in the paper
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
    exploration_after_stagnation: int = 10  # Force exploration after N stagnation rounds
    tabu_length: int = 8  # Tabu list size
    # PPO / training hyperparameters
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.1
    max_grad_norm: float = 0.5
    low_conflict_improvement_bonus: float = 3.0
    near_feasible_solve_bonus: float = 5.0
    last_violation_bonus: float = 25.0

    def __post_init__(self):
        self.lns = rws_lns(
            instance=self.instance,
            incumbent=self.schedule
        )
        self.destroy_names = list(self.destroy_operators.keys())
        self.repair_names = list(self.repair_operators.keys())
        # State vector: 7 features (adds objective-distance-to-feasibility).
        self.state_dim = 7
        # ActorCritic now expects n_severity=10, n_temp=50 by default
        self.model = ActorCritic(
            self.state_dim,
            len(self.destroy_names),
            len(self.repair_names),
            n_severity=10,
            n_temp=50,
        )
        self._apply_mab_repair_prior()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        if self.low_conflict_improvement_bonus < 0:
            raise ValueError("low_conflict_improvement_bonus must be >= 0")
        if self.near_feasible_solve_bonus < 0:
            raise ValueError("near_feasible_solve_bonus must be >= 0")
        if self.last_violation_bonus < 0:
            raise ValueError("last_violation_bonus must be >= 0")
        self.best_objective = float("inf")
        self.current_objective = float("inf")
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

    def reset_instance(self, instance: RWS.Instance, schedule: RWS.Schedule) -> None:
        """Reset search state for a new training instance while keeping PPO weights."""
        self.instance = instance
        self.schedule = schedule
        self.lns = rws_lns(instance=instance, incumbent=schedule)
        self.best_objective = float("inf")
        self.current_objective = float("inf")
        self.prev_improved = 0
        self.prev_best = 0
        self.prev_accepted = 0
        self.stagnation = 0
        self.iteration = 0
        self.tabu_history.clear()
        self.tabu_signatures.clear()

    # --------------------------------------------------------
    def _apply_mab_repair_prior(self) -> None:
        """Bias initial repair policy toward fast solvers like MBandit."""
        if not self.repair_names:
            return
        raw_weights = []
        for name in self.repair_names:
            if "fast" in name:
                raw_weights.append(0.35)
            elif "long" in name:
                raw_weights.append(0.15)
            else:
                raw_weights.append(0.15)
        total = sum(raw_weights)
        probs = [weight / total for weight in raw_weights]
        prior_logits = torch.log(
            torch.tensor(
                probs,
                dtype=self.model.repair_head.bias.dtype,
                device=self.model.repair_head.bias.device,
            )
        )
        with torch.no_grad():
            self.model.repair_head.bias.copy_(prior_logits)

    # --------------------------------------------------------
    def _get_state(self):
        """Return 7-element state vector:
        [best_improved, current_accepted, current_improved, is_current_best,
         cost_diff_best, stagnation_count, objective_gap_to_zero]
        """
        # best_improved: whether best improved in last step
        best_improved = float(self.prev_best)
        # current_accepted: whether last candidate was accepted
        current_accepted = float(self.prev_accepted)
        # current_improved: whether last accepted move improved incumbent
        current_improved = float(self.prev_improved)
        # is_current_best / cost_diff_best based on objective values.
        if (not math.isfinite(self.current_objective)) or (not math.isfinite(self.best_objective)):
            is_current_best = 1.0
            cost_diff_best = -1.0
        elif self.current_objective <= self.best_objective:
            is_current_best = 1.0
            cost_diff_best = -1.0
        else:
            is_current_best = 0.0
            cost_diff_best = (self.current_objective - self.best_objective) / max(1.0, self.best_objective)
        stagnation_count = float(self.stagnation)
        if not math.isfinite(self.current_objective):
            objective_gap_to_zero = 1.0
        else:
            objective_gap_to_zero = math.log1p(max(0.0, self.current_objective))

        return np.array([
            best_improved,
            current_accepted,
            current_improved,
            is_current_best,
            cost_diff_best,
            stagnation_count,
            objective_gap_to_zero,
        ], dtype=np.float32)

    # ========== NEW: TABU & EXPLORATION METHODS ==========
    def _destroyed_signature(self, destroyed_pairs: list[tuple[int, int]]) -> tuple:
        """Extract day/worker sets from destroyed `(day, worker)` pairs."""
        days = frozenset(day for day, _ in destroyed_pairs)
        workers = frozenset(worker for _, worker in destroyed_pairs)
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
        step_iteration = self.iteration + 1
        
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
            self.prev_accepted = 0
            self.stagnation += 1
            self.iteration = step_iteration
            return self._get_state(), -1.0, {
                "iteration": step_iteration,
                "destroy_name": destroy_name,
                "repair_name": repair_name,
                "severity": severity,
                "temperature": temperature,
                "incumbent_objective": (
                    self.current_objective if math.isfinite(self.current_objective) else None
                ),
                "contender_objective": None,
                "accepted": False,
                "new_best": False,
                "destroy_failed": True,
                "repair_failed": False,
                "error": f"{type(e).__name__}: {e}",
                "used_exploration": self.use_exploration_phase,
                "stagnation": self.stagnation,
            }
        
        # Execute repair
        try:
            repair_op(self.lns)
            contender = self.lns.contender
        except Exception as e:
            self.prev_accepted = 0
            self.stagnation += 1
            self.iteration = step_iteration
            return self._get_state(), -1.0, {
                "iteration": step_iteration,
                "destroy_name": destroy_name,
                "repair_name": repair_name,
                "severity": severity,
                "temperature": temperature,
                "incumbent_objective": (
                    self.current_objective if math.isfinite(self.current_objective) else None
                ),
                "contender_objective": None,
                "accepted": False,
                "new_best": False,
                "destroy_failed": False,
                "repair_failed": True,
                "error": f"{type(e).__name__}: {e}",
                "used_exploration": self.use_exploration_phase,
                "stagnation": self.stagnation,
            }
        
        if contender is None:
            # record that last candidate was not accepted
            self.prev_accepted = 0
            self.stagnation += 1
            self.iteration = step_iteration
            return self._get_state(), -1.0, {
                "iteration": step_iteration,
                "destroy_name": destroy_name,
                "repair_name": repair_name,
                "severity": severity,
                "temperature": temperature,
                "incumbent_objective": (
                    self.current_objective if math.isfinite(self.current_objective) else None
                ),
                "contender_objective": None,
                "accepted": False,
                "new_best": False,
                "destroy_failed": False,
                "repair_failed": True,
                "error": "repair operator did not produce contender",
                "used_exploration": self.use_exploration_phase,
                "stagnation": self.stagnation,
            }
        
        # Evaluate move using MiniZinc objective only.
        contender_objective_raw = getattr(self.lns, "contender_objective", None)
        if contender_objective_raw is None:
            self.prev_accepted = 0
            self.stagnation += 1
            self.iteration = step_iteration
            return self._get_state(), -1.0, {
                "iteration": step_iteration,
                "destroy_name": destroy_name,
                "repair_name": repair_name,
                "severity": severity,
                "temperature": temperature,
                "incumbent_objective": (
                    self.current_objective if math.isfinite(self.current_objective) else None
                ),
                "contender_objective": None,
                "accepted": False,
                "new_best": False,
                "destroy_failed": False,
                "repair_failed": True,
                "error": "repair operator did not return objective",
                "used_exploration": self.use_exploration_phase,
                "stagnation": self.stagnation,
            }
        new_obj = float(contender_objective_raw)
        old_obj = self.current_objective
        has_old_obj = math.isfinite(old_obj)
        delta = new_obj - old_obj if has_old_obj else float("-inf")
        
        # Simulated annealing acceptance
        accepted = False
        acceptance_prob = 0.0
        if (not has_old_obj) or delta <= 0:
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
            self.current_objective = new_obj

            # Primary reward: relative improvement (normalized by baseline)
            if has_old_obj:
                baseline_penalty = old_obj / 10.0
                reward = -(new_obj - old_obj) / max(1.0, baseline_penalty)  # Reward improvement
            # Extra bonus in low-objective regions.
            reward += self.low_conflict_improvement_bonus / (1.0 + max(0.0, new_obj))
            if has_old_obj and new_obj < old_obj and old_obj < 5.0:
                reward += self.near_feasible_solve_bonus * (old_obj - new_obj)
            if has_old_obj and old_obj > 0.0 and new_obj <= 0.0:
                reward += self.last_violation_bonus

            # Distinguish between (a) improvement but not a new global best (small bonus)
            # and (b) new global best (larger bonus).
            if (not has_old_obj) or (new_obj < old_obj):
                improved = 1
                if new_obj < self.best_objective:
                    # New global best
                    self.best_objective = new_obj
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
        if step_iteration % self.tabu_length == 0:
            self._update_tabu_after_eviction()
        
        self.prev_improved = improved
        self.prev_best = new_best
        self.iteration = step_iteration
        
        cumulative_reward += reward
        return self._get_state(), reward, {
            "iteration": step_iteration,
            "destroy_name": destroy_name,
            "repair_name": repair_name,
            "severity": severity,
            "temperature": temperature,
            "incumbent_objective": old_obj if has_old_obj else None,
            "contender_objective": new_obj,
            "accepted": accepted,
            "new_best": bool(new_best),
            "destroy_failed": False,
            "repair_failed": False,
            "error": None,
            "used_exploration": self.use_exploration_phase,
            "stagnation": self.stagnation,
        }

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

    # ------------------------------------------------------
    def train(
        self,
        total_steps: int = 2000,
        instance_paths: list[Path] = (),
        per_instance_cap: int = 500,
        timeout_seconds: float | None = None,
        checkpoint_path: str | Path | None = None,
    ):
        if total_steps <= 0:
            raise ValueError("total_steps must be > 0")
        if per_instance_cap <= 0:
            raise ValueError("per_instance_cap must be > 0")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0 when provided")
        training_paths = list(instance_paths)
        if not training_paths:
            raise ValueError("instance_paths must not be empty")

        log = {
            "destroy": [],
            "repair": [],
            "severity": [],
            "temperature": [],
            "reward": [],
            "current_objective": [],
            "best_objective": [],
            "stagnation": [],
            "episode_return": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "accepted_count": [],
            "cumulative_rewards": [],
            "rollout_length": self.rollout_length,
        }

        loop_start = perf_counter()
        ANSI_GREEN = "\033[32m"
        ANSI_PURPLE = "\033[35m"
        ANSI_RESET = "\033[0m"
        print(
            f"training=drl_ppo total_steps={total_steps} rollout_length={self.rollout_length} "
            f"instances={len(training_paths)} per_instance_cap={per_instance_cap}"
        )

        global_steps = 0
        epoch = 0

        try:
            while global_steps < total_steps:
                for instance_path in training_paths:
                    if global_steps >= total_steps:
                        break

                    instance, schedule = load_instance_and_schedule(
                        file_path=instance_path,
                        cyclicity=True,
                        initial_schedule="random",
                    )
                    self.reset_instance(instance=instance, schedule=schedule)
                    print(f"instance_start={instance_path.name}")

                    state = self._get_state()
                    instance_steps = 0

                    while (
                        global_steps < total_steps
                        and instance_steps < per_instance_cap
                        and self.best_objective > 0.0
                        and self.stagnation < 20
                    ):
                        if timeout_seconds is not None and (perf_counter() - loop_start) >= timeout_seconds:
                            print(
                                f"training_stop=timeout elapsed={perf_counter() - loop_start:.3f}s "
                                f"timeout_seconds={timeout_seconds}"
                            )
                            break
                        rollout_steps = min(
                            self.rollout_length,
                            per_instance_cap - instance_steps,
                            total_steps - global_steps,
                        )

                        states_list = []
                        actions_discrete_list = []
                        rewards = []
                        log_probs = []
                        values = []

                        cumulative_reward = 0.0
                        accepted_count = 0

                        for _ in range(rollout_steps):
                            a_d, a_r, a_sev, a_temp, log_p, value = self._select_action(state)

                            step_start = perf_counter()
                            next_state, reward, step_info = self.step(a_d, a_r, a_sev, a_temp, cumulative_reward)
                            step_runtime = perf_counter() - step_start
                            elapsed_total = perf_counter() - loop_start
                            if timeout_seconds is not None and elapsed_total >= timeout_seconds:
                                print(
                                    f"training_stop=timeout elapsed={elapsed_total:.3f}s "
                                    f"timeout_seconds={timeout_seconds}"
                                )
                                global_steps = total_steps
                                break

                            if step_info["accepted"]:
                                accepted_count += 1
                            cumulative_reward += reward

                            summary_line = (
                                f"step={global_steps + 1} "
                                f"iter={step_info['iteration']} "
                                f"time={elapsed_total:.3f}s "
                                f"step_runtime={step_runtime:.3f}s "
                                f"destroy={step_info['destroy_name']} "
                                f"repair={step_info['repair_name']} "
                                f"objective={step_info['incumbent_objective']}->{step_info['contender_objective']} "
                                f"reward={reward:+.3f} "
                                f"accepted={step_info['accepted']} "
                                f"used_exploration={step_info['used_exploration']} "
                                f"stagnation={step_info['stagnation']}"
                            )
                            if step_info["error"] is not None:
                                summary_line += " repair_failed"
                            if step_info["new_best"]:
                                print(f"{ANSI_GREEN}{summary_line}{ANSI_RESET}")
                            elif step_info["used_exploration"]:
                                print(f"{ANSI_PURPLE}{summary_line}{ANSI_RESET}")
                            else:
                                print(summary_line)

                            log["destroy"].append(step_info["destroy_name"])
                            log["repair"].append(step_info["repair_name"])
                            log["severity"].append(step_info["severity"])
                            log["temperature"].append(step_info["temperature"])
                            log["reward"].append(reward)
                            log["current_objective"].append(self.current_objective)
                            log["best_objective"].append(self.best_objective)
                            log["stagnation"].append(self.stagnation)

                            states_list.append(state)
                            actions_discrete_list.append([a_d, a_r, a_sev, a_temp])
                            rewards.append(reward)
                            log_probs.append(log_p)
                            values.append(value)

                            state = next_state
                            global_steps += 1
                            instance_steps += 1

                        if timeout_seconds is not None and (perf_counter() - loop_start) >= timeout_seconds:
                            break
                        episode_return = sum(rewards)
                        acceptance_rate = accepted_count / max(1, rollout_steps) * 100

                        log["episode_return"].append(episode_return)
                        log["accepted_count"].append(accepted_count)
                        log["cumulative_rewards"].append(cumulative_reward)

                        states = torch.from_numpy(np.array(states_list, dtype=np.float32))
                        actions_discrete = torch.tensor(actions_discrete_list, dtype=torch.long)
                        old_log_probs = torch.stack(log_probs).detach()
                        values = torch.stack(values)

                        values_t = values.detach().squeeze()
                        T = len(rewards)

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
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                        returns_normalized = (returns - returns.mean()) / (returns.std() + 1e-8)

                        pl, vl, ent = self._ppo_update(
                            states,
                            actions_discrete,
                            old_log_probs,
                            returns_normalized,
                            advantages
                        )

                        epoch += 1
                        log["policy_loss"].append(pl)
                        log["value_loss"].append(vl)
                        log["entropy"].append(ent)
                        print(
                            f"epoch={epoch} "
                            f"global_steps={global_steps}/{total_steps} "
                            f"instance_steps={instance_steps}/{per_instance_cap} "
                            f"episode_return={episode_return:+.3f} "
                            f"acceptance_rate={acceptance_rate:.1f}% "
                            f"best_objective={self.best_objective if math.isfinite(self.best_objective) else None} "
                            f"current_objective={self.current_objective if math.isfinite(self.current_objective) else None} "
                            f"policy_loss={pl:.6f} value_loss={vl:.6f} entropy={ent:.6f}"
                        )

                        self.metrics_monitor.record(
                            epoch=epoch,
                            best_objective=self.best_objective,
                            current_objective=self.current_objective,
                            episode_return=episode_return,
                            cumulative_reward=cumulative_reward,
                            acceptance_rate=acceptance_rate,
                            policy_loss=pl,
                            value_loss=vl,
                            entropy=ent,
                            stagnation=self.stagnation
                        )
                    if timeout_seconds is not None and (perf_counter() - loop_start) >= timeout_seconds:
                        break

                    if self.best_objective <= 0.0:
                        print("instance_result=solved reason=objective_zero")
                if timeout_seconds is not None and (perf_counter() - loop_start) >= timeout_seconds:
                    break

        except KeyboardInterrupt:
            print("\n" + "!"*150)
            print("⚠ TRAINING INTERRUPTED BY USER (Ctrl+C)")
            print("Returning best-so-far solution.")
            print("!"*150)

        finally:

            print(
                f"training_completed=True best_objective="
                f"{self.best_objective if math.isfinite(self.best_objective) else None}"
            )

            # ======= Save the result of the training run 
            self.metrics_monitor.save_csv("drl_alns_training_metrics.csv")
            checkpoint = (
                Path(checkpoint_path)
                if checkpoint_path is not None
                else Path(__file__).resolve().parent / "PPO_checkpoints" / f"drl_ppo_checkpoint_{total_steps}.pt"
            )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "destroy_names": self.destroy_names,
                    "repair_names": self.repair_names,
                    "state_dim": self.state_dim,
                    "best_objective": self.best_objective,
                    "current_objective": self.current_objective,
                },
                checkpoint,
            )
            print(f"saved_checkpoint=True path={checkpoint}")

            return self.schedule, log
    
# =========================
# DESTROY / REPAIR HELPERS
# =========================
def _make_repair_operator(model_path: Path, solver_name: str, timeout_seconds: int):
    return _mab_make_repair_operator(model_path, solver_name, timeout_seconds)

def _make_repair_tabu_operator(model_path: Path, solver_name: str, timeout_seconds: int):
    return _mab_make_repair_tabu_operator(model_path, solver_name, timeout_seconds)


def _severity_to_fraction(severity: float) -> float:
    """Convert DRL severity to bounded destroy fraction."""
    return min(1.0, max(0.0, float(severity)))


def _wrap_fraction_destroy(
    factory: Callable[[float], Callable[[rws_lns], list[tuple[int, int]]]]
) -> Callable[[], Callable[[rws_lns, float], list[tuple[int, int]]]]:
    """Wrap MBandit fraction-based destroy factory into DRL severity signature."""
    def _maker() -> Callable[[rws_lns, float], list[tuple[int, int]]]:
        def _op(lns: rws_lns, severity: float) -> list[tuple[int, int]]:
            return factory(_severity_to_fraction(severity))(lns)
        return _op
    return _maker

def _wrap_static_destroy(
    op: Callable[[rws_lns], list[tuple[int, int]]]
) -> Callable[[rws_lns, float], list[tuple[int, int]]]:
    def _wrapped(lns: rws_lns, _severity: float) -> list[tuple[int, int]]:
        return op(lns)
    return _wrapped

# -------------------------
# Destroy Operators (use severity)
# -------------------------
_make_destroy_random_workers = _wrap_fraction_destroy(_mab_make_destroy_random_workers)
_make_destroy_random_days = _wrap_fraction_destroy(_mab_make_destroy_random_days)
_make_destroy_random_window = _wrap_fraction_destroy(_mab_make_destroy_random_window)
_make_destroy_worst_workers = _wrap_fraction_destroy(_mab_make_destroy_worst_workers)
_make_destroy_worst_days = _wrap_fraction_destroy(_mab_make_destroy_worst_days)
def _make_destroy_streak():
    def _op(lns: rws_lns, severity: float) -> list[tuple[int, int]]:
        span = max(1, int(round(1 + 16 * _severity_to_fraction(severity))))
        backward = span // 2
        forward = span - backward - 1
        return _mab_make_destroy_streak(
            worker=None,
            day=None,
            forward=forward,
            backward=backward,
        )(lns)
    return _op


def _make_destroy_worker():
    """Destroy one random worker (severity-agnostic)."""
    def _op(lns: rws_lns, _severity: float) -> list[tuple[int, int]]:
        worker = random.randrange(lns.instance.num_workers)
        return lns.destroy_worker(worker)
    return _op


def _make_destroy_day():
    """Destroy one random day (severity-agnostic)."""
    def _op(lns: rws_lns, _severity: float) -> list[tuple[int, int]]:
        day = random.randrange(lns.instance.num_days)
        return lns.destroy_day(day)
    return _op


def _build_destroy_library(
    instance: RWS.Instance,
    include_legacy: bool = False,
) -> Dict[str, Callable[[rws_lns, float], list[tuple[int, int]]]]:
    ops: Dict[str, Callable[[rws_lns, float], list[tuple[int, int]]]] = {
        "destroy_worker": _make_destroy_worker(),
        "destroy_day": _make_destroy_day(),
        "destroy_worst_workers_10pct": _wrap_static_destroy(_mab_make_destroy_worst_workers(0.10)),
        "destroy_worst_workers_30pct": _wrap_static_destroy(_mab_make_destroy_worst_workers(0.30)),
        "destroy_random_workers_20pct": _wrap_static_destroy(_mab_make_destroy_random_workers(0.20)),
        "destroy_random_window_20pct": _wrap_static_destroy(_mab_make_destroy_random_window(0.20)),
        "destroy_forbidden_sequences_30pct": _wrap_static_destroy(_mab_make_destroy_forbidden_sequences(0.30)),
        "destroy_streak_around_worst_worker": _wrap_static_destroy(
            _mab_make_destroy_streak_around_worst_worker(binomial_p=0.2)
        ),
        "destroy_worst_days_10pct": _wrap_static_destroy(_mab_make_destroy_worst_days(0.10)),
        "destroy_worst_days_30pct": _wrap_static_destroy(_mab_make_destroy_worst_days(0.30)),
        "destroy_random_days_20pct": _wrap_static_destroy(_mab_make_destroy_random_days(0.20)),
    }
    for name, op in _mab_make_destroy_max_border_operators(instance).items():
        ops[name] = _wrap_static_destroy(op)
    if include_legacy:
        ops.update(
            {
                "destroy_worst_workers": _make_destroy_worst_workers(),
                "destroy_random_workers": _make_destroy_random_workers(),
                "destroy_worst_days": _make_destroy_worst_days(),
                "destroy_random_days": _make_destroy_random_days(),
                "destroy_random_window": _make_destroy_random_window(),
                "destroy_streak": _make_destroy_streak(),
            }
        )
    return ops

def _smooth(x, k=30):
    if len(x) < k:
        return x
    import numpy as np
    return np.convolve(x, np.ones(k)/k, mode="valid")


def _instance_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.stem)
    if match:
        return (int(match.group(1)), path.stem)
    return (10**9, path.stem)


def _json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def write_training_log(log, output_path: Path) -> None:
    """Persist training-series data used by DRL plotting."""
    payload = _json_safe(
        {
            "rollout_length": log.get("rollout_length"),
            "episode_return": log.get("episode_return", []),
            "best_objective": log.get("best_objective", []),
            "current_objective": log.get("current_objective", []),
            "policy_loss": log.get("policy_loss", []),
            "value_loss": log.get("value_loss", []),
            "entropy": log.get("entropy", []),
            "accepted_count": log.get("accepted_count", []),
            "stagnation": log.get("stagnation", []),
            "reward": log.get("reward", []),
            "cumulative_rewards": log.get("cumulative_rewards", []),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote training logfile: {output_path}")


def plot_cumulative_reward(log, show: bool = False, output_path: Path | None = None):
    """Plot cumulative reward normalized to 0..100%."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4.5))
    rewards = log.get("reward", [])
    if rewards:
        cumulative = np.cumsum(np.array(rewards, dtype=np.float64))
    else:
        cumulative = np.array(
            log.get("cumulative_rewards", log.get("cumulative_reward", [])),
            dtype=np.float64,
        )

    if cumulative.size > 0:
        cmin = float(np.min(cumulative))
        cmax = float(np.max(cumulative))
        if cmax > cmin:
            normalized = (cumulative - cmin) / (cmax - cmin) * 100.0
        else:
            normalized = np.full_like(cumulative, 100.0)
        steps = np.arange(1, len(normalized) + 1)
        plt.plot(
            steps,
            normalized,
            color="tab:blue",
            linewidth=2.0,
            label="Cumulative reward (normalized)",
        )
        plt.xlabel("Step")
        plt.ylabel("Cumulative reward (%)")
        plt.ylim([0, 100])
        plt.legend()
    else:
        plt.text(
            0.5,
            0.5,
            "No reward data available",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
    plt.title("Cumulative Reward Across Training (Normalized)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"Wrote cumulative reward plot: {output_path}")
    if show:
        plt.show()
    else:
        plt.close()

def plot_training(log, show: bool = False, output_path: Path | None = None):
    #Plot training metrics: episode return, objective, losses, entropy, acceptance rate.
    import matplotlib.pyplot as plt
    def _legend_if_present() -> None:
        handles, labels = plt.gca().get_legend_handles_labels()
        if labels:
            plt.legend()

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
    _legend_if_present()
    plt.grid(True, alpha=0.3)
    
    # -----------------------
    # 2) Best & Current Objective
    # -----------------------
    plt.subplot(2, 3, 2)
    epochs = None
    if "best_objective" in log and log["best_objective"]:
        epochs = list(range(len(log["best_objective"])))
        plt.plot(epochs, log["best_objective"], marker='o', label="Best", linewidth=2)
    if "current_objective" in log and log["current_objective"]:
        if epochs is None:
            epochs = list(range(len(log["current_objective"])))
        plt.plot(epochs, log["current_objective"], marker='s', label="Current", linewidth=2, alpha=0.7)
    plt.title("Objective Over Time", fontsize=12, fontweight='bold')
    plt.xlabel("Step")
    plt.ylabel("MiniZinc Objective")
    _legend_if_present()
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
    _legend_if_present()
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
    _legend_if_present()
    plt.grid(True, alpha=0.3)
    
    # -----------------------
    # 5) Acceptance Rate
    # -----------------------
    plt.subplot(2, 3, 5)
    if "accepted_count" in log and log["accepted_count"]:
        rollout_length = int(log.get("rollout_length", 32))
        rollout_length = max(1, rollout_length)
        acceptance_rates = [count / rollout_length * 100 for count in log["accepted_count"]]
        smoothed_ar = _smooth(acceptance_rates, 5)
        plt.plot(smoothed_ar, linewidth=2, label="Acceptance Rate (smoothed)")
        plt.plot(acceptance_rates, alpha=0.3, label="Raw")
        plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, label="Target ~50%")
    plt.title("Move Acceptance Rate", fontsize=12, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Acceptance Rate (%)")
    plt.ylim([0, 100])
    _legend_if_present()
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
    _legend_if_present()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150)
        print(f"Wrote training plot: {output_path}")
    if show:
        plt.show()
    else:
        plt.close()

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    instance_paths = sorted(
        (base / "Instances1-50").glob("Example*.txt"),
        key=_instance_sort_key,
    )
    if not instance_paths:
        raise FileNotFoundError("no instance files found in Instances1-50")
    instance_path = instance_paths[0]
    instance, schedule = load_instance_and_schedule(
        file_path=instance_path,
        cyclicity=True
    )
    print(f"\n=== Training across {len(instance_paths)} instances ===")
    model_path = base / "rws_instance.mzn"
    repair_ops = {
        "repair_chuffed_fast": _make_repair_operator(model_path, "chuffed", 3),
        "repair_gecode_fast": _make_repair_operator(model_path, "gecode", 3),
        "repair_chuffed_long": _make_repair_operator(model_path, "chuffed", 15),
        "repair_gecode_long": _make_repair_operator(model_path, "gecode", 15),
        "repair_tabu_fast": _make_repair_tabu_operator(model_path, "chuffed", 3),
        "repair_tabu_long": _make_repair_tabu_operator(model_path, "chuffed", 12),
    }
    destroy_ops = _build_destroy_library(instance=instance)
    solver = drl_alns(
        instance=instance,
        schedule=schedule,
        destroy_operators=destroy_ops,
        repair_operators=repair_ops,
    )
    
    try:
        final_schedule, log = solver.train(
            total_steps=500,
            instance_paths=instance_paths,
            per_instance_cap=125,
        )
    except KeyboardInterrupt:
        print("Interrupted during training call.")
        final_schedule = solver.schedule
        log = {}

    print("\nFinal schedule:")
    final_schedule.display_schedule()
    final_schedule.display_validity()

    if log:
        plot_training(log, show=False, output_path=base / "drl-training.png")
        plot_cumulative_reward(log, show=False, output_path=base / "drl-crwd.png")
        write_training_log(log, output_path=base / "logs" / "drl-training.log")
