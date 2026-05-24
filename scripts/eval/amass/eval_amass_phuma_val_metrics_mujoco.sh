#!/bin/bash
# Evaluate AMASS checkpoint on phuma_val.txt using MuJoCo Warp,
# producing both success-rate and motion-tracking-metric reports.
#
# Usage:
#   bash scripts/eval/amass/eval_amass_phuma_val_metrics_mujoco.sh

source scripts/source_mujoco_setup.sh

CHECKPOINT="./pretrained/amass_g1_29dof/model_16000.pt"
MOTION_DIR="./g1_npz/PHUMA_processed"
SPLIT_FILE="./split/phuma_val.txt"
NUM_ENVS=512
MAX_MOTIONS=7557  # = num lines in phuma_val.txt; bounds pool_size to the eval set

python src/holosoma/holosoma/eval_success_rate.py \
    --checkpoint="${CHECKPOINT}" \
    --motion_dir="${MOTION_DIR}" \
    --split_file="${SPLIT_FILE}" \
    --num_envs="${NUM_ENVS}" \
    --max_motions="${MAX_MOTIONS}" \
    --simulator=mjwarp \
    --save_motion_metrics
