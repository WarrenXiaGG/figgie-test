import gymnasium
import gym_examples
import argparse
import random
import numpy as np
import math

action_lookup = np.array([0,1,2,4,8,16])

def get_action_from_expected_value(pb, ps, cur_bids, cur_offers, bidders, offerers, agentid):
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
            p = math.ceil(random.uniform(0.6*pb[i], pb[i]))
            # p = random.uniform(pb[i] * 0.8, pb[i] * 1.1)
            up_price = int(p - cur_bids[i])
            if cur_offers[i] <= p:
                action[8+i] = 1
            elif up_price > 0 or bidders[agentid][i] == 1:
                # # find the best action to approx price p
                # action[i] = min(np.searchsorted(action_lookup, up_price), 5)
                action[i] = p
        else:
            # sell
            p = math.floor(random.uniform(ps[i], ps[i] * 1.4))
            # p = random.uniform(ps[i] * 0.9, ps[i] * 1.1)
            down_price = int(cur_offers[i] - p)
            if cur_bids[i] >= p:
                action[12+i] = 1
            elif down_price > 0 or offerers[agentid][i] == 1:
                # action[4+i] = min(np.searchsorted(action_lookup, down_price), 5)
                action[4+i] = p
    return action


class NoiseTrader:
    def __init__(self, init_obs):
        self.is_discrete = False
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
        action = self.get_action_from_expected_value(pb, pb, obs["bids"], obs["offers"])  
        return action
    
    def get_action_from_expected_value(self, pb, ps, cur_bids, cur_offers):
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
                p = math.ceil(random.uniform(0.6*pb[i], pb[i]))
                # p = random.uniform(pb[i] * 0.8, pb[i] * 1.1)
                up_price = int(p - cur_bids[i])
                if cur_offers[i] <= p:
                    action[8+i] = 1
                elif up_price > 0:
                    # # find the best action to approx price p
                    # action[i] = min(np.searchsorted(action_lookup, up_price), 5)
                    action[i] = p
            else:
                # sell
                p = math.floor(random.uniform(ps[i], ps[i] * 1.4))
                # p = random.uniform(ps[i] * 0.9, ps[i] * 1.1)
                down_price = int(cur_offers[i] - p)
                if cur_bids[i] >= p:
                    action[12+i] = 1
                elif down_price > 0:
                    # action[4+i] = min(np.searchsorted(action_lookup, down_price), 5)
                    action[4+i] = p
        if random.random() < 0.2:
            dec = random.random()
            if dec < 0.5:
                dec = random.random()
                suit = random.randint(0,3)
                if dec < 0.5:
                    action[suit+8] = 1
                else:
                    action[suit+12] = 1
        return action


class RandomTrader:
    def __init__(self, init_obs):
        # self.is_discrete = True
        self.is_discrete = False
        self.sigma = 1

    def get_action(self, obs, info):
        dec = random.random()
        action = np.zeros((16), dtype=int)
        if dec < 0.5:
            dec = random.random()
            suit = random.randint(0,3)
            if dec < 0.25:
                action[8+suit] = 1
            elif dec < 0.5:
                action[12+suit] = 1
            elif dec < 0.75:
                # act = random.randint(0,5)
                act = random.randint(1,10)
                action[suit] = act
            else:
                # act = random.randint(0,5)
                act = random.randint(1,10)
                action[suit+4] = act
        return action

# each line represents a possible deck setting.
# [spades, clubs, hearts, diamonds], majority card number, goal suit payoff
DECK = [
    ([8,12,10,10],5,120),
    ([10,12,8,10],6,100),
    ([10,12,10,8],6,100),
    ([12,8,10,10],5,120),
    ([12,10,8,10],6,100),
    ([12,10,10,8],6,100),
    ([8,10,10,12],6,100),
    ([10,8,10,12],6,100),
    ([10,10,8,12],5,120),
    ([8,10,12,10],6,100),
    ([10,8,12,10],6,100),
    ([10,10,12,8],5,120),
]

class Fundamentalist:
    def __init__(self, num_agents, agentid):
        self.r = 1.3 # 
        self.is_discrete = False
        self.num_agents = num_agents
        self.processed_transaction_id = 0
        self.agentid = agentid

    def deck_likelihood(self, cardcounts, cards):
        '''
        Calc deck likelihood distribution.
        '''
        tot_card_seen = [0] * 4
        for i in range(4):
            for j in range(self.num_agents):
                if j == self.agentid:
                    tot_card_seen[i] += cards[i]
                else:
                    tot_card_seen[i] += cardcounts[j][i]
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
                tmp = self.get_eb(j, n[j] - 1, m)
                if tmp >= 8: # a threshold for selling
                    ps.append(50)
                else:
                    ps.append(tmp)
        return pb, ps

    def delete_outdated_transactions(self):
        '''
        TODO: all orders currently in the market that no longer agree with the agent's 
        expected values are cleared by way of lazy deletion.
        '''
        pass

    def get_action(self, obs, info):
        m = self.deck_likelihood(obs["cardcounts"], obs["cards"])
        pb, ps = self.estimate_trade_value(m, obs["cards"])
        action = get_action_from_expected_value(pb, ps, obs["bids"], obs["offers"], obs["bidder"], obs["offerer"], self.agentid)  
        # self.delete_outdated_transactions()
        return action

if __name__ == "__main__":
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

    # Note: this following part is wrong now since the figgie step() function changed

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
