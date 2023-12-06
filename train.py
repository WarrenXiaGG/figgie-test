from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium
from baseline import NoiseTrader,Fundamentalist,RandomTrader
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--agents', nargs='+', default=["P", "N", "N", "N", "N"])
parser.add_argument('--model_name', type=str, default="./models/ppo_figgie")
parser.add_argument('--skip_train', action='store_true')
parser.add_argument('--skip_eval', action='store_true')
parser.add_argument('--train_step', type=int, default=2000000)
parser.add_argument('--eval_epoch' , type=int, default=20)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

assert len(args.agents) == 5 and args.agents[0] == "P"


agents = []
for i in range(5):
    if args.agents[i] == "P":
        agents.append("ppo")
    elif args.agents[i] == "N":
        agents.append(NoiseTrader([]))
    elif args.agents[i] == "R":
        agents.append(RandomTrader([]))
    elif args.agents[i] == "F":
        agents.append(Fundamentalist({'cards':[0,0,0,0]}, 5))
    else:
        raise NotImplementedError

env = gymnasium.make("gym_examples/Figgie-v0",agents=agents, output_debug_info=args.debug)
wrapped_env = gymnasium.wrappers.RecordEpisodeStatistics(env, 50)

if args.skip_train == True:
    model = PPO.load(args.model_name)
else:
    model = PPO("MultiInputPolicy", wrapped_env, verbose=1)
    # Note from xzy: in order to use progress_bar, install 'tqdm' and 'rich' packages
    model.learn(total_timesteps=args.train_step, progress_bar=True)
    model.save(args.model_name)

if args.skip_eval == False:
    # Using packages to evaluate the average Reward of the trained agent
    eval_results = evaluate_policy(model, env, n_eval_episodes=args.eval_epoch, render=False, return_episode_rewards=True)
    print(eval_results)
    total_rewards = eval_results[0]
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")

    # Evaluate the average earned money
    delta_money = []
    for epoch in range(args.eval_epoch):
        obs, info = wrapped_env.reset()
        terminated = False
        while True:
            action, _ = model.predict(obs)
            obs, rewards, terminated, truncated, info = wrapped_env.step(action)
            if terminated:
                delta_money.append(obs['money'][0] - 500)
                break
    print(delta_money)
    mean_money = np.mean(delta_money)
    std_money = np.std(delta_money)
    print(f"Mean Money: {mean_money}, Std Money: {std_money}")


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
