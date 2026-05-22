source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-phuma-future-motion \
    logger:wandb \
    --logger.entity draftrec \
    --logger.project holosoma \
    --logger.name=g1_29dof_amass_future_diverse_action_scale_finetune \
    --logger.video.enabled False \
    --training.checkpoint="logs/holosoma/20260502_g1_29dof_amass_future_diverse_action_scale/model_16000.pt" \
    --training.finetune True \
    --command.setup_terms.motion_command.params.motion_config.motion_dir="./g1_npz/AMASS_processed/humanml/train" \
    --algo.config.eval_callbacks.success_rate.val_split_file=./split/phuma_val.txt \
    --algo.config.eval_callbacks.success_rate.val_motion_dir=./g1_npz/PHUMA_processed \
    --algo.config.save_interval 1000 \
    --algo.config.eval_interval 1000 \
    robot:g1-29dof-diverse-action-scale
