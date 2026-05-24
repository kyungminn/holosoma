#!/bin/bash
# Evaluate AMASS checkpoint on phuma_val.txt using IsaacSim,
# producing both success-rate and motion-tracking-metric reports.
#
# Usage:
#   bash scripts/eval/amass/eval_amass_phuma_val_metrics_isaacsim.sh

source scripts/source_isaacsim_setup.sh

CHECKPOINT="./pretrained/amass_g1_29dof/model_16000.pt"
MOTION_DIR="./g1_npz/PHUMA_processed"
SPLIT_FILE="./split/phuma_val.txt"
NUM_ENVS=7557
MAX_MOTIONS=7557

python src/holosoma/holosoma/eval_success_rate.py \
    --checkpoint="${CHECKPOINT}" \
    --motion_dir="${MOTION_DIR}" \
    --split_file="${SPLIT_FILE}" \
    --num_envs="${NUM_ENVS}" \
    --max_motions="${MAX_MOTIONS}" \
    --save_motion_metrics
