#!/bin/zsh

# Added "LunarLander" at the end to ensure we only pick up LunarLander folders
runscripts/eval_claix.sh box2d_lunar_lander_ppo claix_cpu ppo_cpu_hybrid results/smac_hand_crafted LunarLander
#runscripts/eval_claix.sh box2d_lunar_lander_ppo claix_cpu ppo_cpu_hybrid results/smac_mf LunarLander
runscripts/eval_claix.sh box2d_lunar_lander_ppo claix_cpu ppo_cpu_hybrid results/smac_rf LunarLander
runscripts/eval_claix.sh box2d_lunar_lander_ppo claix_cpu ppo_cpu_hybrid results/smac LunarLander
#runscripts/eval_claix.sh box2d_lunar_lander_ppo claix_cpu ppo_cpu_hybrid results/rs LunarLander



runscripts/eval_claix.sh box2d_lunar_lander_ppo claix_cpu ppo_cpu_algo results/smac_hand_crafted LunarLander
runscripts/eval_claix.sh box2d_lunar_lander_ppo claix_cpu ppo_cpu_algo results/smac_rf LunarLander
runscripts/eval_claix.sh box2d_lunar_lander_ppo claix_cpu ppo_cpu_algo results/smac LunarLander
#runscripts/eval_claix.sh box2d_lunar_lander_ppo claix_cpu ppo_cpu_algo results/smac_mf LunarLander
#runscripts/eval_claix.sh box2d_lunar_lander_ppo claix_cpu ppo_cpu_algo results/rs LunarLander

