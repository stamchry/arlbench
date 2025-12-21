#!/bin/zsh

# Added "ant" at the end to ensure we only pick up ant folders
runscripts/eval_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_hybrid results/smac_hand_crafted ant
runscripts/eval_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_hybrid results/smac_mf ant
runscripts/eval_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_hybrid results/smac_rf ant
runscripts/eval_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_hybrid results/smac ant
runscripts/eval_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_hybrid results/rs ant



runscripts/eval_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_algo results/smac_hand_crafted ant
runscripts/eval_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_algo results/smac_rf ant
runscripts/eval_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_algo results/smac ant
runscripts/eval_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_algo results/smac_mf ant
runscripts/eval_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_algo results/rs ant