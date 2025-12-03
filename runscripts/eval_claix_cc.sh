#!/bin/zsh

# Added "CartPole" at the end to ensure we only pick up CartPole folders
#runscripts/eval_claix.sh cc_cartpole_ppo claix_cpu ppo_cpu_hybrid results/smac_hand_crafted CartPole
#runscripts/eval_claix.sh cc_cartpole_ppo claix_cpu ppo_cpu_hybrid results/rs CartPole

runscripts/eval_claix.sh box2d_lunar_lander_ppo claix_gpu_h100 ppo_gpu_hybrid results/rs LunarLander-v2