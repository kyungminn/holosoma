"""PBHC-style RFI DR variant of g1_29dof_phuma_future_motion.

Same as g1_29dof_phuma_future_motion, but swaps the randomization preset to
g1_29dof_wbt_randomization_with_rfi which enables torque RFI noise injection
(rfi_lim=0.1) and per-env RFI scale randomization ([0.5, 1.5]) on top of the
existing PD-gain DR.
"""

from dataclasses import replace

from holosoma.config_values.wbt.g1.multi_motion_experiment import g1_29dof_phuma_future_motion
from holosoma.config_values.wbt.g1.randomization import g1_29dof_wbt_randomization_with_rfi

g1_29dof_phuma_future_motion_rfi = replace(
    g1_29dof_phuma_future_motion,
    training=replace(
        g1_29dof_phuma_future_motion.training,
        name="g1_29dof_phuma_future_motion_rfi",
    ),
    randomization=g1_29dof_wbt_randomization_with_rfi,
)

__all__ = ["g1_29dof_phuma_future_motion_rfi"]
