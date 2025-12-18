#!/bin/zsh

# Added "CartPole" at the end to ensure we only pick up CartPole folders
runscripts/eval_claix.sh cc_cartpole_ppo claix_cpu ppo_cpu_hybrid results/smac_hand_crafted CartPole
runscripts/eval_claix.sh cc_cartpole_ppo claix_cpu ppo_cpu_hybrid results/smac_mf CartPole
runscripts/eval_claix.sh cc_cartpole_ppo claix_cpu ppo_cpu_hybrid results/smac_rf CartPole
runscripts/eval_claix.sh cc_cartpole_ppo claix_cpu ppo_cpu_hybrid results/smac CartPole
runscripts/eval_claix.sh cc_cartpole_ppo claix_cpu ppo_cpu_hybrid results/rs CartPole


runscripts/eval_claix.sh cc_cartpole_ppo claix_cpu ppo_cpu_algo results/smac CartPole
runscripts/eval_claix.sh cc_cartpole_ppo claix_cpu ppo_cpu_algo results/smac_mf CartPole
runscripts/eval_claix.sh cc_cartpole_ppo claix_cpu ppo_cpu_algo results/rs CartPole

