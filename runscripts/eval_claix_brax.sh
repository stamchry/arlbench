#!/bin/zsh

# Random Search
runscripts/eval_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_hybrid results/rs ant
runscripts/eval_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_algo results/rs ant

# Base SMAC
runscripts/eval_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_hybrid results/smac ant
runscripts/eval_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_algo results/smac ant

# Cost-Aware (RF)
runscripts/eval_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_hybrid results/smac_rf ant
runscripts/eval_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_algo results/smac_rf ant

# Cost-Aware (GP)
runscripts/eval_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_hybrid results/smac_gp ant
runscripts/eval_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_algo results/smac_gp ant

# Cost-Aware (Symbolic)
runscripts/eval_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_hybrid results/smac_symbolic_gpu ant
runscripts/eval_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_algo results/smac_symbolic_gpu ant


# Ablation Study - Initial Design
runscripts/eval_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_hybrid results/smac_rf_ablation ant
runscripts/eval_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_algo results/smac_rf_ablation ant

runscripts/eval_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_hybrid results/smac_gp_ablation ant
runscripts/eval_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_algo results/smac_gp_ablation ant

runscripts/eval_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_hybrid results/smac_symbolic_gpu_ablation ant
runscripts/eval_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_algo results/smac_symbolic_gpu_ablation ant

# Ablation Study - Acquisition Function
runscripts/eval_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_hybrid results/smac_rf_ei_ablation ant
runscripts/eval_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_algo results/smac_rf_ei_ablation ant

runscripts/eval_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_hybrid results/smac_gp_ei_ablation ant
runscripts/eval_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_algo results/smac_gp_ei_ablation ant

runscripts/eval_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_hybrid results/smac_symbolic_gpu_ei_ablation ant
runscripts/eval_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_algo results/smac_symbolic_gpu_ei_ablation ant

