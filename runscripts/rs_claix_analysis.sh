#!/bin/zsh

# GPU Environments
runscripts/rs_claix.sh minatar_breakout_ppo claix_gpu_h100 ppo_gpu_hybrid tune_rs
runscripts/rs_claix.sh minatar_space_invaders_ppo claix_gpu_h100 ppo_gpu_hybrid tune_rs
runscripts/rs_claix.sh brax_halfcheetah_ppo claix_gpu_h100 ppo_gpu_hybrid tune_rs
runscripts/rs_claix.sh brax_humanoid_ppo claix_gpu_h100 ppo_gpu_hybrid tune_rs

# CPU Environments
runscripts/rs_claix.sh cc_acrobot_ppo claix_cpu ppo_cpu_hybrid tune_rs
runscripts/rs_claix.sh cc_pendulum_ppo claix_cpu ppo_cpu_hybrid tune_rs
runscripts/rs_claix.sh cc_mountain_car_ppo claix_cpu ppo_cpu_hybrid tune_rs

