source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-wbt-future-motion \
    --task.model-path ./pretrained/sim_to_real_compare/phuma/model_85000.onnx \
    --task.motion-file-path ./selected_motions/Sim_to_Real_test/stationary/reach/humanml_002551_chunk_0000.npz \
    --task.use-joystick \
    --task.rl-rate 50 \
    --task.interface eth0 \
    --task.save-metrics \
    "$@"
