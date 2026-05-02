#!/bin/zsh

# Random Search
runscripts/eval_claix.sh cc_cartpole_ppo claix_cpu ppo_cpu_hybrid results/rs CartPole
runscripts/eval_claix.sh cc_cartpole_ppo claix_cpu ppo_cpu_algo results/rs CartPole

# Base SMAC
runscripts/eval_claix.sh cc_cartpole_ppo claix_cpu ppo_cpu_hybrid results/smac CartPole
runscripts/eval_claix.sh cc_cartpole_ppo claix_cpu ppo_cpu_algo results/smac CartPole

# Cost-Aware (RF)
runscripts/eval_claix.sh cc_cartpole_ppo claix_cpu ppo_cpu_hybrid results/smac_rf CartPole
runscripts/eval_claix.sh cc_cartpole_ppo claix_cpu ppo_cpu_algo results/smac_rf CartPole

# Cost-Aware (GP)
runscripts/eval_claix.sh cc_cartpole_ppo claix_cpu ppo_cpu_hybrid results/smac_gp CartPole
runscripts/eval_claix.sh cc_cartpole_ppo claix_cpu ppo_cpu_algo results/smac_gp CartPole

# Cost-Aware (Symbolic)
runscripts/eval_claix.sh cc_cartpole_ppo claix_cpu ppo_cpu_hybrid results/smac_symbolic_cpu CartPole
runscripts/eval_claix.sh cc_cartpole_ppo claix_cpu ppo_cpu_algo results/smac_symbolic_cpu CartPole

# Ablation Study - Initial Design
runscripts/eval_claix.sh cc_cartpole_ppo claix_cpu ppo_cpu_hybrid results/smac_rf_ablation CartPole
runscripts/eval_claix.sh cc_cartpole_ppo claix_cpu ppo_cpu_algo results/smac_rf_ablation CartPole

runscripts/eval_claix.sh cc_cartpole_ppo claix_cpu ppo_cpu_hybrid results/smac_gp_ablation CartPole
runscripts/eval_claix.sh cc_cartpole_ppo claix_cpu ppo_cpu_algo results/smac_gp_ablation CartPole

runscripts/eval_claix.sh cc_cartpole_ppo claix_cpu ppo_cpu_hybrid results/smac_symbolic_cpu_ablation CartPole
runscripts/eval_claix.sh cc_cartpole_ppo claix_cpu ppo_cpu_algo results/smac_symbolic_cpu_ablation CartPole

# Ablation Study - Acquisition Function
runscripts/eval_claix.sh cc_cartpole_ppo claix_cpu ppo_cpu_hybrid results/smac_rf_ei_ablation CartPole
runscripts/eval_claix.sh cc_cartpole_ppo claix_cpu ppo_cpu_algo results/smac_rf_ei_ablation CartPole

runscripts/eval_claix.sh cc_cartpole_ppo claix_cpu ppo_cpu_hybrid results/smac_gp_ei_ablation CartPole
runscripts/eval_claix.sh cc_cartpole_ppo claix_cpu ppo_cpu_algo results/smac_gp_ei_ablation CartPole

runscripts/eval_claix.sh cc_cartpole_ppo claix_cpu ppo_cpu_hybrid results/smac_symbolic_cpu_ei_ablation CartPole
runscripts/eval_claix.sh cc_cartpole_ppo claix_cpu ppo_cpu_algo results/smac_symbolic_cpu_ei_ablation CartPole