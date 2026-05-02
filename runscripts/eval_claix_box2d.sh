#!/bin/zsh

# Random Search
runscripts/eval_claix.sh box2d_lunar_lander_ppo claix_cpu ppo_cpu_hybrid results/rs LunarLander
runscripts/eval_claix.sh box2d_lunar_lander_ppo claix_cpu ppo_cpu_algo results/rs LunarLander

# Base SMAC
runscripts/eval_claix.sh box2d_lunar_lander_ppo claix_cpu ppo_cpu_hybrid results/smac LunarLander
runscripts/eval_claix.sh box2d_lunar_lander_ppo claix_cpu ppo_cpu_algo results/smac LunarLander

# Cost-Aware (RF)
runscripts/eval_claix.sh box2d_lunar_lander_ppo claix_cpu ppo_cpu_hybrid results/smac_rf LunarLander
runscripts/eval_claix.sh box2d_lunar_lander_ppo claix_cpu ppo_cpu_algo results/smac_rf LunarLander

# Cost-Aware (GP)
runscripts/eval_claix.sh box2d_lunar_lander_ppo claix_cpu ppo_cpu_hybrid results/smac_gp LunarLander
runscripts/eval_claix.sh box2d_lunar_lander_ppo claix_cpu ppo_cpu_algo results/smac_gp LunarLander

# Cost-Aware (Symbolic)
runscripts/eval_claix.sh box2d_lunar_lander_ppo claix_cpu ppo_cpu_hybrid results/smac_symbolic_cpu LunarLander
runscripts/eval_claix.sh box2d_lunar_lander_ppo claix_cpu ppo_cpu_algo results/smac_symbolic_cpu LunarLander


# Ablation Study - Initial Design
runscripts/eval_claix.sh box2d_lunar_lander_ppo claix_cpu ppo_cpu_hybrid results/smac_rf_ablation LunarLander
runscripts/eval_claix.sh box2d_lunar_lander_ppo claix_cpu ppo_cpu_algo results/smac_rf_ablation LunarLander

runscripts/eval_claix.sh box2d_lunar_lander_ppo claix_cpu ppo_cpu_hybrid results/smac_gp_ablation LunarLander
runscripts/eval_claix.sh box2d_lunar_lander_ppo claix_cpu ppo_cpu_algo results/smac_gp_ablation LunarLander

runscripts/eval_claix.sh box2d_lunar_lander_ppo claix_cpu ppo_cpu_hybrid results/smac_symbolic_cpu_ablation LunarLander
runscripts/eval_claix.sh box2d_lunar_lander_ppo claix_cpu ppo_cpu_algo results/smac_symbolic_cpu_ablation LunarLander

# Ablation Study - Acquisition Function
runscripts/eval_claix.sh box2d_lunar_lander_ppo claix_cpu ppo_cpu_hybrid results/smac_rf_ei_ablation LunarLander
runscripts/eval_claix.sh box2d_lunar_lander_ppo claix_cpu ppo_cpu_algo results/smac_rf_ei_ablation LunarLander

runscripts/eval_claix.sh box2d_lunar_lander_ppo claix_cpu ppo_cpu_hybrid results/smac_gp_ei_ablation LunarLander
runscripts/eval_claix.sh box2d_lunar_lander_ppo claix_cpu ppo_cpu_algo results/smac_gp_ei_ablation LunarLander

runscripts/eval_claix.sh box2d_lunar_lander_ppo claix_cpu ppo_cpu_hybrid results/smac_symbolic_cpu_ei_ablation LunarLander
runscripts/eval_claix.sh box2d_lunar_lander_ppo claix_cpu ppo_cpu_algo results/smac_symbolic_cpu_ei_ablation LunarLander