"""Tuned variant of g1_29dof_phuma_future_motion.

Selectively picks training improvements from upstream PR #76 while keeping the
sim-to-real-relevant settings of the local branch intact:

Applied from upstream:
    * Reward weight relaxation (limits_dof_pos, undesired_contacts) via
      `g1_29dof_wbt_reward_tuned`.
    * Initial root-velocity noise increased to match upstream values
      (root_lin_vel ~5x, root_ang_vel ~5x).
    * `randomize_dof_state.randomize_dof_pos_bias` disabled at reset.
      This is the reset-time per-episode joint-pos jitter — distinct from the
      env-lifetime `setup_dof_pos_bias` (which we keep). The reset variant is
      effectively redundant with the existing motion start-frame noise and was
      flagged with a TODO in the original code suggesting it was unintentional.
    * PPO learning rate bumped from 1e-5 (PPOConfig default) to 1e-3 for both
      actor and critic, matching upstream's BeyondMimic-style settings. The
      schedule is adaptive (default), so KL will pull the LR down as needed.
    * `num_learning_epochs` reduced 8 -> 5 to pair with the higher LR
      (fewer epochs per update offset by larger per-step effective steps).
    * Periodic eval (`eval_callbacks.success_rate`) now runs against
      `./split/phuma_val.txt` instead of the training split. The callback
      transparently swaps the motion library's file list to the val subset
      for the eval and reloads training motions afterwards.

NOT applied (intentional):
    * PD-gain DR (`actuator_randomizer_state.enable_pd_gain`) is left ON.
      Upstream PR #76 disabled this to lift PPO sim metrics; that change is
      not validated for sim-to-real, and BeyondMimic itself ships with PD-gain
      randomization. Helps cover IsaacSim<->MuJoCo PD-controller differences.
    * `setup_dof_pos_bias` (env-lifetime default-dof-pos shift) is left ON to
      simulate per-robot encoder calibration drift.
    * motion_ends termination is left ON. This branch pairs it with a
      default-pose handoff at motion end and depends on it.
    * BadTracking termination is unchanged (this branch's BadTracking is
      already z-axis-only, matching upstream's BadTrackingZOnly behavior).
"""

from dataclasses import replace

from pydantic.dataclasses import dataclass as pydantic_dataclass

from holosoma.config_types.command import NoiseToInitialPoseConfig
from holosoma.config_values.wbt.g1.multi_motion_experiment import (
    _build_command_cfg,
    g1_29dof_phuma_future_motion,
)
from holosoma.config_values.wbt.g1.reward_tuned import g1_29dof_wbt_reward_tuned


@pydantic_dataclass(frozen=True)
class _SuccessRateCbValConfig:
    """Eval callback config: runs SuccessRateCallback against the val split.

    `val_motion_dir` is left empty by default so the callback falls back to
    `motion_library.motion_dir` (i.e. the CLI-overridden training motion_dir).
    """

    _target_: str = "holosoma.agents.callbacks.success_rate_callback.SuccessRateCallback"
    val_split_file: str = "./split/phuma_val.txt"
    val_motion_dir: str = ""


_eval_callbacks_val = {"success_rate": _SuccessRateCbValConfig()}

_init_pose_config_tuned = NoiseToInitialPoseConfig(
    overall_noise_scale=1.0,
    dof_pos=0.1,
    root_pos=[0.05, 0.05, 0.01],
    root_rot=[0.1, 0.1, 0.2],
    root_lin_vel=[0.5, 0.5, 0.2],
    root_ang_vel=[0.52, 0.52, 0.78],
    object_pos=[0.0, 0.0, 0.0],
)

_base_motion_cfg = (
    g1_29dof_phuma_future_motion.command.setup_terms["motion_command"].params["motion_config"]
)
_tuned_motion_cfg = replace(_base_motion_cfg, noise_to_initial_pose=_init_pose_config_tuned)

# Disable reset-time `randomize_dof_pos_bias` only; keep setup-time
# `setup_dof_pos_bias` and all other DR knobs untouched.
_base_rand = g1_29dof_phuma_future_motion.randomization
_base_reset_dof = _base_rand.reset_terms["randomize_dof_state"]
_tuned_reset_dof = replace(
    _base_reset_dof,
    params={**_base_reset_dof.params, "randomize_dof_pos_bias": False},
)
_tuned_randomization = replace(
    _base_rand,
    reset_terms={**_base_rand.reset_terms, "randomize_dof_state": _tuned_reset_dof},
)

# Bump actor/critic LR from 1e-5 -> 1e-3 (adaptive schedule will damp from KL)
# and shorten num_learning_epochs 8 -> 5 to match upstream PR #76.
# Also point the periodic eval at the val split rather than the training set.
_base_algo_cfg = g1_29dof_phuma_future_motion.algo
_tuned_algo = replace(
    _base_algo_cfg,
    config=replace(
        _base_algo_cfg.config,
        actor_learning_rate=1e-3,
        critic_learning_rate=1e-3,
        num_learning_epochs=5,
        eval_callbacks=_eval_callbacks_val,
    ),
)

g1_29dof_phuma_future_motion_tuned = replace(
    g1_29dof_phuma_future_motion,
    training=replace(
        g1_29dof_phuma_future_motion.training,
        name="g1_29dof_phuma_future_motion_tuned",
    ),
    algo=_tuned_algo,
    command=_build_command_cfg(_tuned_motion_cfg),
    reward=g1_29dof_wbt_reward_tuned,
    randomization=_tuned_randomization,
)

__all__ = ["g1_29dof_phuma_future_motion_tuned"]
