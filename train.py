
import gymnasium

import gym_examples

env = gymnasium.make("gym_examples/Figgie-v0")
wrapped_env = gymnasium.wrappers.RecordEpisodeStatistics(env, 50)


obs,info = wrapped_env.reset()
print(obs)

for _ in range(500):
    action = wrapped_env.action_space.sample()
    observation, reward, terminated, truncated, info = wrapped_env.step(action)
    if terminated:
        wrapped_env.end_round()
        break
    #print(observation)
