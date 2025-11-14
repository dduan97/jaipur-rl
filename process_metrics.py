import pickle
import numpy as np
import pandas as pd

records = []
dir = "/home/ubuntu/cs230/checkpoints/20251113/"
for i in range(35):
    path = f"{dir}/step_{i}/result_info.pkl"
    with open(path, 'rb') as f:
        loaded_data = pickle.load(f)
    total_loss = loaded_data["learner"]["player_policy"]["learner_stats"]["total_loss"]
    vf_loss = loaded_data["learner"]["player_policy"]["learner_stats"]["vf_loss"]
    policy_loss = loaded_data["learner"]["player_policy"]["learner_stats"]["policy_loss"]
    step = i
    records.append({
        "step": step,
        "total_loss": total_loss,
        "vf_loss": vf_loss,
        "policy_loss": policy_loss
    })

df = pd.DataFrame(records)
df.to_csv(f"{dir}/training_metrics.csv", index=False)