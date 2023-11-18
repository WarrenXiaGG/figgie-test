import gymnasium
import gym_examples
import argparse
import random
import numpy as np
import math


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

# each line represents a possible deck setting.
# [spades, clubs, hearts, diamonds], majority card number, goal suit payoff
DECK = [
    ([12,8,10,10],5,120),
    ([12,10,8,10],6,100),
    ([12,10,10,8],6,100),
    ([8,12,10,10],5,120),
    ([10,12,8,10],6,100),
    ([10,12,10,8],6,100),
    ([8,10,12,10],6,100),
    ([10,8,12,10],6,100),
    ([10,10,12,8],5,120),
    ([8,10,10,12],6,100),
    ([10,8,10,12],6,100),
    ([10,10,8,12],5,120),
]

class Fundamentalist:
    def __init__(self, init_obs, num_agents, agentid):
        self.r = 2.0 # 
        self.num_agents = num_agents
        self.agentid = agentid
        self.processed_transaction_id = 0
        # initialize card counting
        self.L = np.zeros((4, self.num_agents), dtype=int)
        for i in range(4):
            self.L[i][self.agentid] = init_obs['cards'][i]

    def card_counting(self, transactions):
        for i in range(self.processed_transaction_id, len(transactions), 1):
            # update for the i-th transaction
            seller, buyer, color, money = transactions[i]
            self.L[color][buyer] += 1
            self.L[color][seller] = max(self.L[color][seller] - 1, 0)
        self.processed_transaction_id = len(transactions)

    def deck_likelihood(self):
        '''
        Calc deck likelihood distribution.
        TODO: still has bug here!! ValueError
        '''
        tot_card_seen = [0] * 4
        for i in range(4):
            for j in range(self.num_agents):
                tot_card_seen[i] += self.L[i][j]
        comb = np.ones(12) # possible combinations of each deck
        for j, (cards, majority, payoff) in enumerate(DECK):
            for i in range(4):
                if cards[i] < tot_card_seen[i]:
                    comb[j] = 0.0
                    break
                comb[j] *= math.comb(cards[i], tot_card_seen[i])
        prob_deck = comb / np.sum(comb)
        return prob_deck

    def get_eb(self, j, jn, m):
        '''
        Returns the expected values for buying color j, given that
        the agent currently possesses jn cards of color j,
        and the prob dist. for the deck in play is m.
        '''
        val = 0.0
        # consider the deck setting that color j is goal suit
        for i in range(3*j, 3*j+3, 1):
            majority = DECK[i][1]
            payout = DECK[i][2]
            if jn >= majority:
                val_maj = 0
            else:
                val_maj = payout * (1 - self.r) / (1 - self.r**majority) * (self.r**jn)
            val += m[i] * (10 + val_maj)
        return val

    def estimate_trade_value(self, m, n):
        '''
        Given m (size 12, probability distribution of decks), 
            n (size 4, # of cards agent currently possesses),
        Returns the estimated buy and sell value of each card color
        '''
        pb = []
        ps = []
        for j in range(4):
            pb.append(self.get_eb(j, n[j], m))
            if n[j] == 0:
                ps.append(None)
            else:
                ps.append(self.get_eb(j, n[j] - 1, m))
        return pb, ps

    def delete_outdated_transactions(self):
        '''
        TODO: all orders currently in the market that no longer agree with the agent's 
        expected values are cleared by way of lazy deletion.
        '''
        pass

    def get_action(self, obs, info):
        self.card_counting(info["transaction_history"])
        m = self.deck_likelihood()
        pb, ps = self.estimate_trade_value(m, obs["cards"])
        action = get_action_from_expected_value(pb, ps, obs["bids"], obs["offers"])  
        # self.delete_outdated_transactions()
        return action

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=["noise", "fundamental", "bottom"], default="noise")
args = parser.parse_args()


env = gymnasium.make("gym_examples/Figgie-v0")
wrapped_env = gymnasium.wrappers.RecordEpisodeStatistics(env, 50)

obs,info = wrapped_env.reset()
action_lookup = wrapped_env.get_wrapper_attr('action_lookup')
num_agents = wrapped_env.get_wrapper_attr('num_agents')

if args.mode == "noise":
    policy = NoiseTrader(obs)
elif args.mode == "fundamental":
    policy = Fundamentalist(obs, num_agents, agentid=0)
else:
    raise NotImplementedError

curr_player = 0
terminated = False

for _ in range(500):
    # player0: NoiseTrader, player1-4: random
    if curr_player > 0:
        action = wrapped_env.action_space.sample()
        _, _, terminated, truncated, _ = wrapped_env.step(action)
    else:
        action = policy.get_action(obs, info)
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
    if terminated:
        wrapped_env.end_round()
        break
    curr_player = (curr_player + 1) % 5
    # print(observation)
