import numpy as np

from environment import jaipur

env = jaipur.JaipurEnv()
env.reset()

for agent in env.agent_iter():
    obs_and_mask = env.observe(agent)
    action_mask = obs_and_mask["action_mask"]
    act = np.random.choice([i for i in range(len(action_mask)) if action_mask[i]])
    env.step(act)
    env.render()
    if all(env.terminations.values()):
        break