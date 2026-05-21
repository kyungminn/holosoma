source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-phuma-future-motion-tuned \
    logger:wandb \
    --logger.entity draftrec \
    --logger.project holosoma \
    --logger.name=g1_29dof_phuma_future_diverse_action_scale_tuned \
    --logger.video.enabled False \
    --training.checkpoint="pretrained/phuma_g1_29dof_new/model_104000.pt" \
    --training.finetune True \
    --algo.config.load_optimizer False \
    --command.setup_terms.motion_command.params.motion_config.motion_dir="./g1_npz/PHUMA_processed" \
    --command.setup_terms.motion_command.params.motion_config.split_file="./split/phuma_train.txt" \
    robot:g1-29dof-diverse-action-scale
