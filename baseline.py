import gymnasium
import gym_examples
import argparse
import random
import numpy as np


def get_action_from_expected_value(pb, ps, cur_bids, cur_offers):
    '''
    parameters:
    pb: expected value for buying cards
    ps: expected value for selling cards
    cur_offers & cur_bids: current price
    hb: lowest sell order price in the history (not considering for now)
    hs: highest buy order price in the history (not considering for now)
    returns:
        action array: penny_up, penny_down, buy, sell (4*4)
    '''
    action = np.zeros(16, dtype=int)
    for i in range(4):
        if pb[i] is None or ps[i] is None:
            continue
        # randomly decide to buy or sell
        if random.choice([0,1]) == 1:
            # buy
            p = random.uniform(0, pb[i])
            up_price = int(p - cur_bids[i])
            action[8+i] = 1
            if up_price > 0:
                # find the best action to approx price p
                action[i] = np.argmin(np.abs(action_lookup - up_price))
        else:
            # sell
            p = random.uniform(ps[i], ps[i] * 2)
            down_price = int(cur_offers[i] - p)
            action[12+i] = 1
            if down_price > 0:
                action[4+i] = np.argmin(np.abs(action_lookup - down_price))
    return action

class NoiseTrader:
    def __init__(self, init_obs):
        self.sigma = 1

    def get_action(self, obs, info):
        transaction_hitory = info["transaction_history"]
        highest_buy = [None] * 4
        for transaction in transaction_hitory:
            _, _, color, money = transaction
            if highest_buy[color] is None or highest_buy[color] < money:
                highest_buy[color] = money
        pb = highest_buy
        for i in range(4):
            if pb[i] is None:
                continue
            else:
                pb[i] *= np.exp(np.random.normal(0, self.sigma))
        action = get_action_from_expected_value(pb, pb, obs["bids"], obs["offers"])  
        return action

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=["noise", "fundamental"], default="noise")
args = parser.parse_args()


env = gymnasium.make("gym_examples/Figgie-v0")
wrapped_env = gymnasium.wrappers.RecordEpisodeStatistics(env, 50)

obs,info = wrapped_env.reset()
action_lookup = wrapped_env.get_wrapper_attr('action_lookup')
num_agents = wrapped_env.get_wrapper_attr('num_agents')

if args.mode == "noise":
    policy = NoiseTrader(obs)
else:
    raise NotImplementedError

curr_player = 0

for _ in range(500):
    # player4: NoiseTrader, player0-3: random
    if curr_player < 4:
        action = wrapped_env.action_space.sample()
    else:
        action = policy.get_action(obs, info)
    obs, reward, terminated, truncated, info = wrapped_env.step(action)
    if terminated:
        wrapped_env.end_round()
        break
    curr_player = (curr_player + 1) % 5
    #print(observation)
