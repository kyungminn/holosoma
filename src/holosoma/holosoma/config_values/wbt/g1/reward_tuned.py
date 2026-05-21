"""Tuned WBT reward variants for G1.

Picks the reward-weight relaxations from upstream PR #76 (BeyondMimic-style
training tuning) without changing DR or termination. Specifically:
    * limits_dof_pos:    -100.0  -> -10.0
    * undesired_contacts: -0.5   -> -0.1
penalty_dof_acc is left at its default (weight 0; toggle via CLI override).
"""

from dataclasses import replace

from holosoma.config_values.wbt.g1.reward import g1_29dof_wbt_reward

g1_29dof_wbt_reward_tuned = replace(
    g1_29dof_wbt_reward,
    terms={
        **g1_29dof_wbt_reward.terms,
        "limits_dof_pos": replace(g1_29dof_wbt_reward.terms["limits_dof_pos"], weight=-10.0),
        "undesired_contacts": replace(g1_29dof_wbt_reward.terms["undesired_contacts"], weight=-0.1),
    },
)

__all__ = ["g1_29dof_wbt_reward_tuned"]
