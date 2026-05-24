#!/bin/bash
# Evaluate PhUMA checkpoint on the unseen_video split using IsaacSim,
# producing both success-rate and motion-tracking-metric reports.
#
# Usage:
#   bash scripts/eval/phuma/eval_phuma_unseen_video_isaacsim.sh

source scripts/source_isaacsim_setup.sh

CHECKPOINT="./pretrained/phuma_g1_29dof_new/model_104000.pt"
MOTION_DIR="./g1_npz/PHUMA_processed"
SPLIT_FILE="./split/unseen_video.txt"
NUM_ENVS=512
MAX_MOTIONS=504  # = num lines in unseen_video.txt; bounds pool_size to the eval set

python src/holosoma/holosoma/eval_success_rate.py \
    --checkpoint="${CHECKPOINT}" \
    --motion_dir="${MOTION_DIR}" \
    --split_file="${SPLIT_FILE}" \
    --num_envs="${NUM_ENVS}" \
    --max_motions="${MAX_MOTIONS}" \
    --save_motion_metrics
