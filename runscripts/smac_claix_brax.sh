#!/bin/zsh

#runscripts/smac_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_hybrid tune_smac_cost_aware_hand
#runscripts/smac_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_hybrid tune_smac_cost_aware_rf
#runscripts/smac_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_hybrid tune_smac
runscripts/smac_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_hybrid tune_smac_mf
#runscripts/rs_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_hybrid tune_rs

#runscripts/smac_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_algo tune_smac_cost_aware_hand
#runscripts/smac_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_algo tune_smac_cost_aware_rf
#runscripts/smac_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_algo tune_smac
runscripts/smac_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_algo tune_smac_mf
#runscripts/rs_claix.sh brax_ant_ppo claix_gpu_h100 ppo_gpu_algo tune_rs

