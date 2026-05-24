"""Motion tracking metric evaluation callback.

Accumulates per-env per-step body position / joint vel / joint accel errors
during the existing batched eval rollout (see eval_success_rate.py) and
writes a summary text report mirroring success_rate_iter{N:05d}.txt.

Designed to coexist with SuccessRateCallback: both share the same
batched rollout. SuccessRateCallback owns batch setup / stop control;
this callback only observes state and accumulates.

Metrics reported:
  - l-mpjpe (mm): root-relative per-body position error, averaged across
    the env's alive frames, then across motions. Source: motion_command's
    pre-computed `metrics["motion/error_body_pos"]` (already root-frame-
    aligned via body_pos_relative_w).
  - dof_vel_err (rad/s): mean over alive frames of ||vel_actual - vel_gt||_2.
  - dof_acc_err (rad/s^2): finite-differenced from dof_vel_err with dt=1/rl_rate.
"""

from __future__ import annotations

import csv
import glob as glob_module
import os
from collections import defaultdict

import torch
from loguru import logger

from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.managers.command.terms.wbt import MotionLoader


class MotionMetricsCallback(RLEvalCallback):
    """Accumulate l-mpjpe / dof vel / dof accel errors during the eval rollout."""

    def __init__(
        self,
        config=None,
        training_loop=None,
        val_split_file: str = "",
        val_motion_dir: str = "",
        max_eval_motions: int = 0,
        rl_rate: float = 50.0,
        sr_callback=None,
    ):
        super().__init__(config, training_loop)
        self._num_envs = training_loop.env.num_envs
        self._val_split_file = val_split_file or ""
        self._val_motion_dir = val_motion_dir or ""
        self._max_eval_motions = max(0, int(max_eval_motions))
        self._rl_rate = float(rl_rate)  # retained for context; ASAP-style metrics ignore dt
        self._sr_callback = sr_callback  # optional: share eval set selection
        self._using_val = False
        self._eval_paths_cache: list[str] | None = None
        self._eval_loaders_cache: list = []

    # ----- Eval set helpers (mirror SuccessRateCallback) -----

    def _build_val_file_list(self, motion_dir: str, split_file: str) -> list[str]:
        all_npz = sorted(glob_module.glob(os.path.join(motion_dir, "**", "*.npz"), recursive=True))
        if not all_npz:
            raise FileNotFoundError(f"MotionMetricsCallback: no .npz files found in {motion_dir}")
        with open(split_file, "r") as f:
            split_keys = set(line.strip() for line in f if line.strip())
        filtered = []
        for npz_path in all_npz:
            rel = os.path.relpath(npz_path, motion_dir)
            key = os.path.splitext(rel)[0]
            if key in split_keys:
                filtered.append(npz_path)
        if not filtered:
            raise FileNotFoundError(
                f"MotionMetricsCallback: split file {split_file} matched 0/{len(all_npz)} .npz under {motion_dir}"
            )
        return filtered

    def _load_metadata(self, split_file: str):
        self._motion_skill: dict[str, str] = {}
        self._motion_category: dict[str, str] = {}
        if not split_file:
            return
        csv_path = os.path.splitext(split_file)[0] + ".csv"
        if not os.path.isfile(csv_path):
            return
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                chunk = row.get("chunk_file", "").strip()
                skill = row.get("skill", "").strip()
                category = row.get("category", "").strip()
                if chunk:
                    self._motion_skill[chunk] = skill
                    self._motion_category[chunk] = category

    def _get_motion_keys(self, npz_path: str) -> list[str]:
        motion_dir = getattr(self._motion_library, "motion_dir", None)
        keys: list[str] = []
        if motion_dir:
            try:
                rel = os.path.relpath(npz_path, motion_dir)
                keys.append(os.path.splitext(rel)[0].replace(os.sep, "_"))
            except ValueError:
                pass
        keys.append(os.path.splitext(os.path.basename(npz_path))[0])
        seen = set()
        unique = []
        for k in keys:
            if k not in seen:
                seen.add(k)
                unique.append(k)
        return unique

    def _lookup_metadata(self, mapping: dict, npz_path: str) -> str | None:
        for k in self._get_motion_keys(npz_path):
            if k in mapping:
                return mapping[k]
        return None

    # ----- Eval lifecycle -----

    def on_pre_evaluate_policy(self):
        env = self.training_loop.env
        mc = env.command_manager.get_state("motion_command")
        if not hasattr(mc, "motion_library"):
            logger.warning("MotionMetricsCallback: no motion_library found, skipping eval")
            self._skip = True
            return
        self._skip = False
        self._env = env
        self._motion_command = mc
        self._motion_library = mc.motion_library

        # Prefer SR callback's view of eval set (if attached) — it has already
        # swapped the motion_library's file list to the eval set at this point.
        if self._sr_callback is not None and getattr(self._sr_callback, "_motion_library", None) is mc.motion_library:
            # Older SR callback versions don't expose `_using_val` — fall back to False.
            self._using_val = bool(getattr(self._sr_callback, "_using_val", False))
            num_motions = self._sr_callback._num_motions
            split_file = self._val_split_file or getattr(mc.motion_cfg, "split_file", "")
        else:
            # Standalone: replicate SR's eval-set selection.
            if self._val_split_file:
                motion_dir = self._val_motion_dir or mc.motion_library.motion_dir
                files = self._build_val_file_list(motion_dir, self._val_split_file)
                mc.motion_library._all_npz_files = files
                self._using_val = True
                split_file = self._val_split_file
            else:
                split_file = getattr(mc.motion_cfg, "split_file", "")
            num_motions = len(mc.motion_library._all_npz_files)
            if self._max_eval_motions > 0 and num_motions > self._max_eval_motions:
                mc.motion_library._all_npz_files = mc.motion_library._all_npz_files[: self._max_eval_motions]
                num_motions = self._max_eval_motions

        self._num_motions = num_motions
        self._load_metadata(split_file)

        # Per-MOTION accumulators (sized to num_motions, not num_envs). Required
        # for splits larger than num_envs, where SuccessRateCallback iterates
        # multiple batches and the same env_id evaluates different motions in
        # successive batches.
        # All three metrics use body positions (root-relative pos for l-mpjpe,
        # world-frame pos for ASAP-style vel/accel finite differences).
        num_bodies = len(mc.motion_cfg.body_names_to_track)
        self._sum_body_err_mm = torch.zeros(self._num_motions, num_bodies, device=self.device)
        self._sum_vel_err_mm = torch.zeros(self._num_motions, num_bodies, device=self.device)
        self._sum_accel_err_mm = torch.zeros(self._num_motions, num_bodies, device=self.device)
        self._step_count = torch.zeros(self._num_motions, dtype=torch.long, device=self.device)
        self._vel_count = torch.zeros(self._num_motions, dtype=torch.long, device=self.device)
        self._accel_count = torch.zeros(self._num_motions, dtype=torch.long, device=self.device)

        # Per-step state for body-pos finite differences (valid only within a batch).
        # Body positions for vel/accel are in world frame so env_origins cancel.
        self._prev_body_pos_pred: torch.Tensor | None = None   # body_pos at t-1
        self._prev_body_pos_gt: torch.Tensor | None = None
        self._prev2_body_pos_pred: torch.Tensor | None = None  # body_pos at t-2 (for accel)
        self._prev2_body_pos_gt: torch.Tensor | None = None
        self._prev_active_mask: torch.Tensor | None = None     # active at t-1
        self._prev2_active_mask: torch.Tensor | None = None    # active at t-2

        # Batch tracking: snapshot at pre_step (because SR may increment
        # _current_batch in its post_step, which runs before mine).
        self._snapshot_batch = 0
        self._snapshot_batch_size = min(self._num_envs, self._num_motions)
        self._my_prev_batch = -1

        self._tracked_body_names = list(mc.motion_cfg.body_names_to_track)

        set_tag = "val" if self._using_val else "train"
        logger.info(
            f"MotionMetricsCallback ({set_tag}): tracking {self._num_motions} motions, "
            f"{num_bodies} bodies (l-mpjpe + ASAP-style body vel/accel errors, mm units)"
        )

    def on_pre_eval_env_step(self, actor_state):
        """Snapshot SR's batch index BEFORE env.step (and before SR's post_step
        possibly advances _current_batch). The env.step that follows uses this
        batch's motion assignments, so we attribute the resulting data to it.
        """
        if getattr(self, "_skip", False):
            return actor_state
        if self._sr_callback is not None and hasattr(self._sr_callback, "_current_batch"):
            self._snapshot_batch = int(self._sr_callback._current_batch)
            self._snapshot_batch_size = int(getattr(self._sr_callback, "_batch_size", self._snapshot_batch_size))
        return actor_state

    def on_post_eval_env_step(self, actor_state):
        if getattr(self, "_skip", False):
            return actor_state

        mc = self._motion_command
        batch_idx = self._snapshot_batch
        batch_size = self._snapshot_batch_size
        batch_start = batch_idx * self._num_envs

        # Batch transition: reset finite-diff state (vel/accel across batches is meaningless).
        if batch_idx != self._my_prev_batch:
            self._prev_body_pos_pred = None
            self._prev_body_pos_gt = None
            self._prev2_body_pos_pred = None
            self._prev2_body_pos_gt = None
            self._prev_active_mask = None
            self._prev2_active_mask = None
            self._my_prev_batch = batch_idx

        # Use SR's _env_done as the per-batch alive mask if available; else assume all alive.
        if self._sr_callback is not None and hasattr(self._sr_callback, "_env_done"):
            env_done = self._sr_callback._env_done[:batch_size]
        else:
            env_done = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        active = ~env_done  # [batch_size]

        # Per-body root-relative position error in mm (l-mpjpe).
        body_err_mm = (
            torch.norm(mc.body_pos_relative_w - mc.robot_body_pos_w, dim=-1) * 1000.0
        )[:batch_size]  # [batch_size, B]

        # Current frame's body positions in world frame, for vel/accel finite-diff.
        # Both have env_origins added → cancel in finite differences.
        curr_body_pos_pred = mc.robot_body_pos_w[:batch_size]  # [batch_size, B, 3]
        curr_body_pos_gt = mc.body_pos_w[:batch_size]          # [batch_size, B, 3]

        # --- Position error (l-mpjpe) accumulation ---
        active_env_ids = active.nonzero(as_tuple=True)[0]  # [K]
        if active_env_ids.numel() > 0:
            motion_ids = active_env_ids + batch_start
            self._sum_body_err_mm.index_add_(0, motion_ids, body_err_mm[active_env_ids])
            self._step_count.index_add_(
                0, motion_ids, torch.ones_like(motion_ids, dtype=self._step_count.dtype)
            )

        # --- ASAP-style velocity error: per-body 1st-order finite diff of body world pos, mm ---
        #     vel_err = || (pred[t] - pred[t-1]) - (gt[t] - gt[t-1]) || * 1000
        # Only when env was active at both t and t-1.
        if (
            self._prev_body_pos_pred is not None
            and self._prev_active_mask is not None
        ):
            vel_active = active & self._prev_active_mask
            vel_env_ids = vel_active.nonzero(as_tuple=True)[0]
            if vel_env_ids.numel() > 0:
                vel_motion_ids = vel_env_ids + batch_start
                vel_diff = (
                    (curr_body_pos_pred[vel_env_ids] - self._prev_body_pos_pred[vel_env_ids])
                    - (curr_body_pos_gt[vel_env_ids] - self._prev_body_pos_gt[vel_env_ids])
                )  # [K2, B, 3]
                vel_err_per_body_mm = torch.norm(vel_diff, dim=-1) * 1000.0  # [K2, B]
                self._sum_vel_err_mm.index_add_(0, vel_motion_ids, vel_err_per_body_mm)
                self._vel_count.index_add_(
                    0, vel_motion_ids,
                    torch.ones_like(vel_motion_ids, dtype=self._vel_count.dtype),
                )

        # --- ASAP-style accel error: 2nd-order finite diff of body world pos, mm ---
        #     accel_err = || (pred[t-2] - 2*pred[t-1] + pred[t]) - (gt[t-2] - 2*gt[t-1] + gt[t]) || * 1000
        # Only when env was active at t, t-1, and t-2.
        if (
            self._prev2_body_pos_pred is not None
            and self._prev_active_mask is not None
            and self._prev2_active_mask is not None
        ):
            accel_active = active & self._prev_active_mask & self._prev2_active_mask
            accel_env_ids = accel_active.nonzero(as_tuple=True)[0]
            if accel_env_ids.numel() > 0:
                accel_motion_ids = accel_env_ids + batch_start
                accel_pred = (
                    self._prev2_body_pos_pred[accel_env_ids]
                    - 2.0 * self._prev_body_pos_pred[accel_env_ids]
                    + curr_body_pos_pred[accel_env_ids]
                )  # [K3, B, 3]
                accel_gt = (
                    self._prev2_body_pos_gt[accel_env_ids]
                    - 2.0 * self._prev_body_pos_gt[accel_env_ids]
                    + curr_body_pos_gt[accel_env_ids]
                )
                accel_err_per_body_mm = torch.norm(accel_pred - accel_gt, dim=-1) * 1000.0  # [K3, B]
                self._sum_accel_err_mm.index_add_(0, accel_motion_ids, accel_err_per_body_mm)
                self._accel_count.index_add_(
                    0, accel_motion_ids,
                    torch.ones_like(accel_motion_ids, dtype=self._accel_count.dtype),
                )

        # Roll buffers forward: t-2 ← t-1, t-1 ← t.
        self._prev2_body_pos_pred = self._prev_body_pos_pred
        self._prev2_body_pos_gt = self._prev_body_pos_gt
        self._prev2_active_mask = self._prev_active_mask
        self._prev_body_pos_pred = curr_body_pos_pred.clone()
        self._prev_body_pos_gt = curr_body_pos_gt.clone()
        self._prev_active_mask = active.clone()
        return actor_state

    def on_post_evaluate_policy(self):
        if getattr(self, "_skip", False):
            return

        num_total = self._num_motions
        cnt = self._step_count.clamp(min=1).to(torch.float32)
        vcnt = self._vel_count.clamp(min=1).to(torch.float32)
        acnt = self._accel_count.clamp(min=1).to(torch.float32)

        per_motion_per_body_mpjpe = self._sum_body_err_mm / cnt[:, None]   # [num_motions, B] mm
        per_motion_per_body_vel   = self._sum_vel_err_mm / vcnt[:, None]   # [num_motions, B] mm
        per_motion_per_body_accel = self._sum_accel_err_mm / acnt[:, None] # [num_motions, B] mm

        # Motions that were never sampled OR whose accumulator got poisoned with
        # NaN (e.g. physics blow-up made body positions NaN) → set per-motion
        # values to NaN, then use nanmean throughout so corrupt motions don't
        # poison aggregate metrics.
        unsampled = self._step_count == 0
        nan_body = torch.isnan(per_motion_per_body_mpjpe).any(dim=-1)
        nan_vel = torch.isnan(per_motion_per_body_vel).any(dim=-1)
        nan_acc = torch.isnan(per_motion_per_body_accel).any(dim=-1)
        broken = unsampled | nan_body | nan_vel | nan_acc
        if broken.any():
            per_motion_per_body_mpjpe[broken] = float("nan")
            per_motion_per_body_vel[broken] = float("nan")
            per_motion_per_body_accel[broken] = float("nan")

        per_motion_mpjpe = per_motion_per_body_mpjpe.mean(dim=-1)        # [num_motions]
        per_motion_vel   = per_motion_per_body_vel.mean(dim=-1)
        per_motion_acc   = per_motion_per_body_accel.mean(dim=-1)
        per_body_mpjpe = torch.nanmean(per_motion_per_body_mpjpe, dim=0)  # [B]
        overall_mpjpe = float(torch.nanmean(per_motion_mpjpe).item())
        overall_vel   = float(torch.nanmean(per_motion_vel).item())
        overall_acc   = float(torch.nanmean(per_motion_acc).item())

        num_broken = int(broken.sum().item())
        num_unsampled = int(unsampled.sum().item())

        # Stash machine-readable results on the callback for callers (e.g. eval script).
        # ASAP convention: vel/accel are body-pos finite differences in mm (no dt division),
        # not true physical units.
        self.metrics = {
            "MotionMetrics/l_mpjpe_mm": overall_mpjpe,
            "MotionMetrics/vel_err_mm": overall_vel,
            "MotionMetrics/accel_err_mm": overall_acc,
            "MotionMetrics/num_motions": float(num_total),
        }
        for name, val in zip(self._tracked_body_names, per_body_mpjpe.tolist()):
            self.metrics[f"MotionMetrics/body_mpjpe_mm/{name}"] = float(val)

        logger.info("=" * 60)
        logger.info("=== Motion Metrics Eval Results (ASAP convention) ===")
        logger.info("=" * 60)
        logger.info(f"  l-mpjpe   (mean over motions):     {overall_mpjpe:.3f} mm")
        logger.info(f"  vel_err   (mean over motions):     {overall_vel:.3f} mm  (1st-order body pos diff)")
        logger.info(f"  accel_err (mean over motions):     {overall_acc:.3f} mm  (2nd-order body pos diff)")

        # Write text report.
        log_dir = getattr(self.training_loop, "log_dir", None)
        if not log_dir:
            return
        iteration = getattr(self.training_loop, "current_learning_iteration", 0)
        suffix = "_val" if self._using_val else ""
        txt_path = os.path.join(str(log_dir), f"motion_metrics_iter{iteration:05d}{suffix}.txt")
        try:
            lib = self._motion_library
            lines = []
            lines.append(f"=== Motion Metrics Report (iteration {iteration}) ===")
            lines.append(f"Eval set: {'val' if self._using_val else 'train'}, num_motions={num_total}")
            if num_broken > 0:
                lines.append(
                    f"WARNING: {num_broken} motions excluded from aggregates "
                    f"({num_unsampled} unsampled, {num_broken - num_unsampled} NaN-poisoned by physics blow-up)"
                )
            lines.append("")
            lines.append("Overall (ASAP convention; vel/accel are body-pos finite diffs in mm, no dt division):")
            lines.append(f"  l-mpjpe   (mean over motions): {overall_mpjpe:.3f} mm")
            lines.append(f"  vel_err   (mean over motions): {overall_vel:.3f} mm  (1st-order body-pos diff)")
            lines.append(f"  accel_err (mean over motions): {overall_acc:.3f} mm  (2nd-order body-pos diff)")
            lines.append("")
            lines.append("--- Per-Body l-mpjpe (mm) ---")
            for name, val in zip(self._tracked_body_names, per_body_mpjpe.tolist()):
                lines.append(f"  {name:30s} {val:8.3f}")
            lines.append("")

            if self._motion_skill:
                # Per-skill and per-category breakdowns
                skill_groups: dict[str, list[int]] = defaultdict(list)
                cat_groups: dict[str, list[int]] = defaultdict(list)
                for i in range(num_total):
                    path = lib._all_npz_files[i]
                    sk = self._lookup_metadata(self._motion_skill, path)
                    if sk:
                        skill_groups[sk].append(i)
                    ca = self._lookup_metadata(self._motion_category, path)
                    if ca:
                        cat_groups[ca].append(i)

                def _group_report(groups, label):
                    out = [f"--- Per-{label} l-mpjpe / vel / accel (all mm, ASAP) ---"]
                    for k in sorted(groups.keys()):
                        idxs = torch.tensor(groups[k], dtype=torch.long, device=self.device)
                        # nanmean to skip unsampled motions
                        mp = float(torch.nanmean(per_motion_mpjpe[idxs]).item())
                        ve = float(torch.nanmean(per_motion_vel[idxs]).item())
                        ac = float(torch.nanmean(per_motion_acc[idxs]).item())
                        out.append(
                            f"  {k:20s} mpjpe={mp:7.2f}  vel={ve:6.3f}  accel={ac:6.3f}  (n={len(groups[k])})"
                        )
                    out.append("")
                    return out

                lines.extend(_group_report(skill_groups, "Skill"))
                lines.extend(_group_report(cat_groups, "Category"))

            lines.append("--- Per-Motion (all mm, ASAP convention) ---")
            mp_list = per_motion_mpjpe.tolist()
            ve_list = per_motion_vel.tolist()
            ac_list = per_motion_acc.tolist()
            sc_list = self._step_count[:num_total].tolist()
            for i in range(num_total):
                lines.append(
                    f"  mpjpe={mp_list[i]:7.2f}  vel={ve_list[i]:6.3f}  accel={ac_list[i]:6.3f}  steps={sc_list[i]:4d}  {lib._all_npz_files[i]}"
                )

            with open(txt_path, "w") as f:
                f.write("\n".join(lines) + "\n")
            logger.info(f"MotionMetricsCallback: saved report to {txt_path}")
        except Exception as e:
            logger.warning(f"MotionMetricsCallback: failed to save report: {e}")
