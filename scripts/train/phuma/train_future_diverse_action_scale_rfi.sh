source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-phuma-future-motion-rfi \
    logger:wandb \
    --logger.entity draftrec \
    --logger.project holosoma \
    --logger.name=g1_29dof_phuma_future_diverse_action_scale_finetune_rfi \
    --logger.video.enabled False \
    --training.checkpoint="pretrained/phuma_g1_29dof_new/model_104000.pt" \
    --training.finetune True \
    --algo.config.eval_interval 1000 \
    --algo.config.eval_callbacks.success_rate.val_split_file=./split/phuma_val.txt \
    --command.setup_terms.motion_command.params.motion_config.motion_dir="./g1_npz/PHUMA_processed" \
    --command.setup_terms.motion_command.params.motion_config.split_file="./split/phuma_train.txt" \
    robot:g1-29dof-diverse-action-scale
