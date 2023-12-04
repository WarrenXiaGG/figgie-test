from stable_baselines3 import PPO
import gymnasium

import gym_examples

from baseline import NoiseTrader,Fundamentalist


noisetrader = NoiseTrader([])
fundamentalist = Fundamentalist({'cards':[0,0,0,0]}, 5, 0)

env = gymnasium.make("gym_examples/Figgie-v0",agents=["ppo",noisetrader,noisetrader,noisetrader,noisetrader])
wrapped_env = gymnasium.wrappers.RecordEpisodeStatistics(env, 50)

model = PPO("MultiInputPolicy", wrapped_env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_figgie")
"""
obs,info = wrapped_env.reset()
print(obs)

for _ in range(500):
    action = wrapped_env.action_space.sample()
    observation, reward, terminated, truncated, info = wrapped_env.step(action)
    if terminated:
        wrapped_env.end_round()
        break
    #print(observation)"""
