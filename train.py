from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium
from baseline import NoiseTrader,Fundamentalist,RandomTrader
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
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

# assert len(args.agents) == 5 and args.agents[0] == "P"


agents = []
for i in range(5):
    if args.agents[i] == "P":
        agents.append("ppo")
    elif args.agents[i] == "N":
        agents.append(NoiseTrader([]))
    elif args.agents[i] == "R":
        agents.append(RandomTrader([]))
    elif args.agents[i] == "F":
        agents.append(Fundamentalist(len(args.agents), i))
    else:
        raise NotImplementedError

env = gymnasium.make("gym_examples/Figgie-v0",agents=agents, output_debug_info=args.debug)
wrapped_env = gymnasium.wrappers.RecordEpisodeStatistics(env, 50)

if args.skip_train == True:
    model = PPO.load(args.model_name)
else:
    model = PPO("MultiInputPolicy", wrapped_env, verbose=1)
    save_interval = 1000
    checkpoint_callback = CheckpointCallback(save_freq=save_interval, save_path=args.model_name+"_checkpoint/", name_prefix='model')
    model.learn(total_timesteps=args.train_step, callback=CallbackList([checkpoint_callback]))
    model.save(args.model_name)


if args.skip_eval == False:

    # Evaluate the average earned money
    delta_money = []
    winner = []
    for epoch in range(args.eval_epoch):
        obs, info = wrapped_env.reset()
        # print(obs)
        terminated = False
        while True:
            action, _ = model.predict(obs)
            # m = agents[0].deck_likelihood(obs['cardcounts'], obs['cards'])
            # x = []
            # for i in range(4):
            #     x.append(0.0)
            #     for j in range(i*3, i*3+3, 1):
            #         x[i] += m[j]
            # print("Deck likelihood: ", x)

            # pb, ps = agents[0].estimate_trade_value(m, obs['cards'])
            # print(pb, ps)
            # import ipdb
            # ipdb.set_trace()

            obs, rewards, terminated, truncated, info = wrapped_env.step(action)
            if terminated:
                winner.append(info['stats']['winner'])
                delta_money.append(obs['money'][0] - 500 - (200/len(args.agents)))
                # print(info['transaction_history'])
                # print(obs['cardcounts'])
                # import ipdb
                # ipdb.set_trace()
                break
    # print(delta_money)
    # print(winner)
    win_rate = [(args.agents[i], winner.count(i) / len(winner) * 100) for i in range(len(args.agents))]
    print(win_rate)
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
