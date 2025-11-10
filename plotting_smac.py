import pandas as pd
import matplotlib.pyplot as plt

results_file_smac = 'results/smac_det_true/dqn_CartPole-v1/0/42/runhistory.csv'
runhistory_smac = pd.read_csv(results_file_smac)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

runhistory_smac.plot(x='config_id', y='performance', kind='line', ax=axes[0], title='Configuration Performance over Time')

# Exclude the last config_id for the cost plot
runhistory_smac.iloc[:-1].plot(
    x='config_id', y='cost', kind='line', ax=axes[1], title='Configuration Cost over Time'
)

plt.tight_layout()
plt.show()