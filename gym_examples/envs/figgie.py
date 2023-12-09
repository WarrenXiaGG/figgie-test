# fmt: off
"""
Make your own custom environment
================================

This documentation overviews creating new environments and relevant
useful wrappers, utilities and tests included in Gymnasium designed for
the creation of new environments. You can clone gym-examples to play
with the code that is presented here. We recommend that you use a virtual environment:

.. code:: console

   git clone https://github.com/Farama-Foundation/gym-examples
   cd gym-examples
   python -m venv .env
   source .env/bin/activate
   pip install -e .

Subclassing gymnasium.Env
-------------------------

Before learning how to create your own environment you should check out
`the documentation of Gymnasium’s API </api/env>`__.

We will be concerned with a subset of gym-examples that looks like this:

.. code:: sh

   gym-examples/
     README.md
     setup.py
     gym_examples/
       __init__.py
       envs/
         __init__.py
         grid_world.py
       wrappers/
         __init__.py
         relative_position.py
         reacher_weighted_reward.py
         discrete_action.py
         clip_reward.py

To illustrate the process of subclassing ``gymnasium.Env``, we will
implement a very simplistic game, called ``GridWorldEnv``. We will write
the code for our custom environment in
``gym-examples/gym_examples/envs/grid_world.py``. The environment
consists of a 2-dimensional square grid of fixed size (specified via the
``size`` parameter during construction). The agent can move vertically
or horizontally between grid cells in each timestep. The goal of the
agent is to navigate to a target on the grid that has been placed
randomly at the beginning of the episode.

-  Observations provide the location of the target and agent.
-  There are 4 actions in our environment, corresponding to the
   movements “right”, “up”, “left”, and “down”.
-  A done signal is issued as soon as the agent has navigated to the
   grid cell where the target is located.
-  Rewards are binary and sparse, meaning that the immediate reward is
   always zero, unless the agent has reached the target, then it is 1.

An episode in this environment (with ``size=5``) might look like this:

where the blue dot is the agent and the red square represents the
target.

Let us look at the source code of ``GridWorldEnv`` piece by piece:
"""

# %%
# Declaration and Initialization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Our custom environment will inherit from the abstract class
# ``gymnasium.Env``. You shouldn’t forget to add the ``metadata``
# attribute to your class. There, you should specify the render-modes that
# are supported by your environment (e.g. ``"human"``, ``"rgb_array"``,
# ``"ansi"``) and the framerate at which your environment should be
# rendered. Every environment should support ``None`` as render-mode; you
# don’t need to add it in the metadata. In ``GridWorldEnv``, we will
# support the modes “rgb_array” and “human” and render at 4 FPS.
#
# The ``__init__`` method of our environment will accept the integer
# ``size``, that determines the size of the square grid. We will set up
# some variables for rendering and define ``self.observation_space`` and
# ``self.action_space``. In our case, observations should provide
# information about the location of the agent and target on the
# 2-dimensional grid. We will choose to represent observations in the form
# of dictionaries with keys ``"agent"`` and ``"target"``. An observation
# may look like ``{"agent": array([1, 0]), "target": array([0, 3])}``.
# Since we have 4 actions in our environment (“right”, “up”, “left”,
# “down”), we will use ``Discrete(4)`` as an action space. Here is the
# declaration of ``GridWorldEnv`` and the implementation of ``__init__``:

import numpy as np
import random

import gymnasium as gym
from gymnasium import spaces


class FiggieEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None,agents=[],round_limit = 100, output_debug_info=False,agentpool=[]):
        self.window_size = 512  # The size of the PyGame window
        self.num_agents = len(agents)
        self.round_limit = round_limit
        self.curr_round = 0
        self.curr_player = 0
        self.money = np.zeros((self.num_agents),dtype='i')
        self.cards = np.zeros((self.num_agents,4),dtype='i')
        self.bids = np.zeros((4),dtype='i')
        self.offers = np.zeros((4),dtype='i')
        self.bidavg = np.zeros((4))
        self.offeravg = np.zeros((4))
        self.bidders = np.zeros((4),dtype='i')
        self.offerers = np.zeros((4),dtype='i')
        self.money_per_agent = 500
        self.offer_limit = 100
        self.bid_limit = 100
        self.action_lookup = np.array([0,1,2,4,8,16])
        self.agents = agents
        self.agentpool = agentpool
        self.card_counts = np.zeros((self.num_agents,4),dtype='i')
        self.output_debug_info = output_debug_info
        self.smoothing = 0.65
        print(agents)

        #Money per agent, buy offers, sell offers, bool array for if you are the one selling or buying, your cards, num cards,you
        self.observation_space = spaces.Dict(
            {
                "money": spaces.Box(0, self.num_agents * self.money_per_agent+1, shape=(self.num_agents,), dtype=int),
                "bids": spaces.Box(0, self.bid_limit+2, shape=(4,), dtype=int),
                "offers": spaces.Box(0, self.offer_limit+2, shape=(4,), dtype=int),
                "bidder": spaces.Box(0, 2, shape=(4,self.num_agents), dtype=int),
                "offerer": spaces.Box(0, 2, shape=(4,self.num_agents), dtype=int),
                "cards": spaces.Box(0, 13, shape=(4,), dtype=int),
                "numcards": spaces.Box(0, 41, shape=(self.num_agents,), dtype=int),
                "you": spaces.Box(0, 2, shape=(self.num_agents,), dtype=int),
                "cardcounts": spaces.Box(0,41,shape=(self.num_agents,4), dtype=int),
                "stepsremaining":spaces.Box(0,1,shape=(1,),dtype=float),
                "bidavg": spaces.Box(0,102,shape=(4,),dtype=float),
                "offeravg": spaces.Box(0,102,shape=(4,),dtype=float),
            }
        )

        # penny up, penny down, buy, sell
        action_size = len(self.action_lookup)
        self.action_space = spaces.MultiDiscrete(np.array([action_size,action_size,action_size,action_size,
                                                           action_size,action_size,action_size,action_size,
                                                           2,2,2,2,
                                                           2,2,2,2]))


        # transaction history
        self.transaction_history = [] # seller, buyer, color, money


        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.font = None

# %%
# Constructing Observations From Environment States
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Since we will need to compute observations both in ``reset`` and
# ``step``, it is often convenient to have a (private) method ``_get_obs``
# that translates the environment’s state into an observation. However,
# this is not mandatory and you may as well compute observations in
# ``reset`` and ``step`` separately:

    def _get_obs(self,agent_id):
        you = np.zeros((self.num_agents),dtype='i')
        you[agent_id] = 1

        numcards = np.sum(self.cards, axis=1).astype(int)
        bidder = np.zeros((4,self.num_agents),dtype='i')
        offerer = np.zeros((4,self.num_agents),dtype='i')

        for i in range(4):
            if self.bidders[i] >= 0:
                bidder[i][self.bidders[i]] = 1
            if self.offerers[i] >= 0:
                offerer[i][self.offerers[i]] = 1
        
        return {"money": self.money, "bids": self.bids, "offers": self.offers, "bidder":bidder, "offerer":offerer,
                "cards": self.cards[agent_id], "numcards": numcards, "you": you,"cardcounts": self.card_counts,
                "stepsremaining":np.array([((self.round_limit-self.curr_round)/self.round_limit)]),
                "bidavg": self.bidavg, "offeravg": self.offeravg}

# %%
# We can also implement a similar method for the auxiliary information
# that is returned by ``step`` and ``reset``. In our case, we would like
# to provide the manhattan distance between the agent and the target:

    def _get_info(self):
        return {
        }
    def end_round(self):
        bonus_winner = np.amax(self.cards[::,self.goal_suit].flatten())
        bonus_winner = np.argwhere(self.cards[::,self.goal_suit].flatten() == bonus_winner).flatten()
        bonus = 200 - self.suit_counts[self.goal_suit]*10
        self.money[bonus_winner] += bonus//len(bonus_winner)
        for agent in range(self.num_agents):
            self.money[agent] += self.cards[agent][self.goal_suit]*10
        if self.output_debug_info:
            print("End of round:",self.money, "Bonus winner:", bonus_winner, "Cards:",self.cards, "Goal suit:",self.goal_suit)
        stats = {'winner':np.argmax(self.money), 'bonus_winner':bonus_winner, 'money':self.money, 'goal_suit':self.goal_suit, 'transaction':self.transaction_history, 'cards':self.cards}
        return stats
    
    def reset_game(self):
        #spades,clubs,diamonds,hearts
        self.bidders.fill(-1)
        self.offerers.fill(-1)
        self.bids.fill(0)
        self.offers.fill(self.offer_limit+1)
        suits = np.array([12,10,10,8],dtype='i')
        goal_suit_lookup = [1,0,3,2]
        np.random.shuffle(suits)
        self.suit_counts = suits
        dealing_array = np.concatenate((np.tile(np.array([0]),self.suit_counts[0]),
        np.tile(np.array([1]),self.suit_counts[1]),
        np.tile(np.array([2]),self.suit_counts[2]),
        np.tile(np.array([3]),self.suit_counts[3]),))
        np.random.shuffle(dealing_array)
        self.cards = np.reshape(dealing_array,(self.num_agents,40//self.num_agents))
        self.cards = np.apply_along_axis(lambda x: np.bincount(x, minlength=4), axis=1, arr=self.cards)
        #print("Dealt",self.cards)
        self.cards = self.cards.astype(int) 
        self.goal_suit = goal_suit_lookup[np.argmax(self.suit_counts)]
        self.original_cards = np.copy(self.cards)
        self.curr_round = 0
        self.curr_player = 0
        self.card_counts.fill(0)
        self.bidavg.fill(0)
        self.offeravg.fill(self.offer_limit+1)
        if self.output_debug_info:
            print("Start Round!")
            print(self.cards)
            print("Goal suit:",self.goal_suit)
        

# %%
# Oftentimes, info will also contain some data that is only available
# inside the ``step`` method (e.g. individual reward terms). In that case,
# we would have to update the dictionary that is returned by ``_get_info``
# in ``step``.

# %%
# Reset
# ~~~~~
#
# The ``reset`` method will be called to initiate a new episode. You may
# assume that the ``step`` method will not be called before ``reset`` has
# been called. Moreover, ``reset`` should be called whenever a done signal
# has been issued. Users may pass the ``seed`` keyword to ``reset`` to
# initialize any random number generator that is used by the environment
# to a deterministic state. It is recommended to use the random number
# generator ``self.np_random`` that is provided by the environment’s base
# class, ``gymnasium.Env``. If you only use this RNG, you do not need to
# worry much about seeding, *but you need to remember to call
# ``super().reset(seed=seed)``* to make sure that ``gymnasium.Env``
# correctly seeds the RNG. Once this is done, we can randomly set the
# state of our environment. In our case, we randomly choose the agent’s
# location and the random sample target positions, until it does not
# coincide with the agent’s position.
#
# The ``reset`` method should return a tuple of the initial observation
# and some auxiliary information. We can use the methods ``_get_obs`` and
# ``_get_info`` that we implemented earlier for that:

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.money.fill(self.money_per_agent)
        self.reset_game()

        observation = self._get_obs(0)
        self.transaction_history = []
        #self.agents = ['ppo']
        #self.agents.extend(random.choices(self.agentpool, k=4))
        #print(observation)
        info = self._get_info()
        info["transaction_history"] = self.transaction_history

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

# %%
# Step
# ~~~~
#
# The ``step`` method usually contains most of the logic of your
# environment. It accepts an ``action``, computes the state of the
# environment after applying that action and returns the 5-tuple
# ``(observation, reward, terminated, truncated, info)``. See
# :meth:`gymnasium.Env.step`. Once the new state of the environment has
# been computed, we can check whether it is a terminal state and we set
# ``done`` accordingly. Since we are using sparse binary rewards in
# ``GridWorldEnv``, computing ``reward`` is trivial once we know
# ``done``.To gather ``observation`` and ``info``, we can again make
# use of ``_get_obs`` and ``_get_info``:
    def takestep(self, action, i, is_discrete=False):
        agentid = i
        
        bids = np.array(action[0:4])
        offers = np.array(action[4:8])
        buy = np.array(action[8:12])
        sell = np.array(action[12:16])

        if is_discrete == True:
            bids = self.bids + self.action_lookup[bids]
            offers = self.offers - self.action_lookup[offers]

        for i in range(4):
            if offers[i] == 0:
                offers[i] = self.offer_limit + 1

        bids = np.clip(bids,0,self.bid_limit)
        offers = np.clip(offers,0,self.offer_limit+1)

        actions_invalid = False

        reward = 0

        #check if we have the cards and the money

        valid_bids = np.nonzero(bids)
        valid_offers = np.nonzero(offers-(self.offer_limit+1))
        enoughmoneytobid = np.sum(bids[valid_bids]) <= self.money[agentid]
        if enoughmoneytobid and np.all(self.cards[agentid][valid_offers]):
            
            for i in range(4):
                if bids[i] == 0:
                    continue
                if self.bidders[i] == agentid or self.bids[i] < bids[i]:
                    self.bidders[i] = agentid
                    self.bids[i] = bids[i]
            for i in range(4):
                if offers[i] == self.offer_limit+1:
                    continue
                if self.offerers[i] == agentid or self.offers[i] > offers[i]:
                    self.offerers[i] = agentid
                    self.offers[i] = offers[i]
            
            # self.bidders[valid_bids] = agentid
            # self.offerers[valid_offers] = agentid
            # self.bids[valid_bids] = bids[valid_bids]
            # self.offers[valid_offers] = offers[valid_offers]

        

        #Penalize invalid moves
        for i in range(4):
            if buy[i] == 1 and (self.offerers[i] == agentid):
                reward -= 0.5
            if sell[i] == 1 and (self.bidders[i] == agentid):
                reward -= 0.5

        #process buy, figgie lets you go into the negatives
        
        #Check if you are buying from yourself and if offers are valid
        for i in range(4):
            if buy[i] == 1 and not (self.offerers[i] == agentid) and not (self.offerers[i] == -1) and self.money[agentid] >= self.offers[i]:
                #print("{} buys {} from {} for ${}".format(agentid,i,self.offerers[i],self.offers[i]))
                self.transaction_history.append([self.offerers[i], agentid, i, self.offers[i]])
                self.offeravg[i] =  self.smoothing*self.offers[i] + (1-self.smoothing) * self.offeravg[i]

                self.card_counts[agentid][i] += 1
                self.card_counts[self.offerers[i]][i] = max(self.card_counts[self.offerers[i]][i] - 1, 0)
                
                self.money[agentid] -= self.offers[i]
                self.money[self.offerers[i]] += self.offers[i]
                #print(self.money)
                self.cards[self.offerers[i]][i] -= 1
                self.cards[agentid][i] += 1
                actions_invalid = True
                self.bidders.fill(-1)
                self.offerers.fill(-1)
                self.bids.fill(0)
                self.offers.fill(self.offer_limit+1)
                break

        #Process sell, figgie lets you go into the negatives

        #Check if you are selling to yourself and if offers are valid
        if not actions_invalid:
            for i in range(4):
                if self.cards[agentid][i] > 0 and sell[i] == 1 and not (self.bidders[i] == agentid) and not (self.bidders[i] == -1):
                    #print("{} sells {} to {} for ${}".format(agentid,i,self.bidders[i],self.bids[i]))
                    self.transaction_history.append([agentid, self.bidders[i], i, self.bids[i]])

                    self.card_counts[self.bidders[i]][i] += 1
                    self.card_counts[agentid][i] = max(self.card_counts[agentid][i] - 1, 0)
                    self.bidavg[i] =  self.smoothing*self.bids[i] + (1-self.smoothing) * self.bidavg[i]
                    
                    self.money[agentid] += self.bids[i]
                    self.money[self.bidders[i]] -= self.bids[i]
                    #print(self.money)
                    self.cards[agentid][i] -= 1
                    self.cards[self.bidders[i]][i] += 1
                    actions_invalid = True
                    self.bidders.fill(-1)
                    self.offerers.fill(-1)
                    self.bids.fill(0)
                    self.offers.fill(self.offer_limit+1)
                    break

        self.curr_player += 1
        if self.curr_player == self.num_agents:
            self.curr_player = 0
            self.curr_round += 1
            
        if self.curr_round == self.round_limit:
            terminated = True
        else:
            terminated = False
        
        observation = self._get_obs(self.curr_player)
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        info["transaction_history"] = self.transaction_history

        return observation, reward, terminated, False, info
    
    def step(self, action):
        observation = self._get_obs(0)
        reward = 0
        terminated = None
        info = None
        
        for i in range(self.num_agents):
            if self.agents[i] == "ppo":
                observation, r, terminated, _ , info = self.takestep(action,i, is_discrete=True)
                reward += r
            else:
                # make sure agents are not cheating by seeing others' observation
                # the info only contains transaction history which is public, so no worries
                observation, _, terminated, _ , info = self.takestep(self.agents[i].get_action(observation,info),i, is_discrete=self.agents[i].is_discrete)

        if terminated == True:
            stats = self.end_round()
            info["stats"] = stats
            #Incentivize getting money
            if np.argmax(self.money) == 0:
                reward += 20
                reward += (self.money[0] - self.money_per_agent)/30
            else:
                reward += (self.money[0]-self.money[np.argmax(self.money)])/4
            #Incentivize getting goal suits
            for suit in range(4):
                delta = self.cards[0][suit]-self.original_cards[0][suit]
                if suit == self.goal_suit:
                    delta *= 0
                else:
                    delta *= -2
                reward += delta
            #if 0 in bonus_winner and (self.money[0] - self.money_per_agent) > 30:
            #    reward += 10
            if self.output_debug_info:
                print(f"End of Round! Money:{self.money[0]}, Last Step Reward:{reward}")
        #print("Bids",observation['bidavg'])
        #print("offers",observation['offeravg'])
        return observation,reward,terminated,False,info
# %%
# Rendering
# ~~~~~~~~~
#
# Here, we are using PyGame for rendering. A similar approach to rendering
# is used in many environments that are included with Gymnasium and you
# can use it as a skeleton for your own environments:

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self.font is None and self.render_mode == "human":
            self.font = pygame.font.Font(pygame.font.get_default_font(), 36)

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

# %%
# Close
# ~~~~~
#
# The ``close`` method should close any open resources that were used by
# the environment. In many cases, you don’t actually have to bother to
# implement this method. However, in our example ``render_mode`` may be
# ``"human"`` and we might need to close the window that has been opened:

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


# %%
# In other environments ``close`` might also close files that were opened
# or release other resources. You shouldn’t interact with the environment
# after having called ``close``.

# %%
# Registering Envs
# ----------------
#
# In order for the custom environments to be detected by Gymnasium, they
# must be registered as follows. We will choose to put this code in
# ``gym-examples/gym_examples/__init__.py``.
#
# .. code:: python
#
#   from gymnasium.envs.registration import register
#
#   register(
#        id="gym_examples/GridWorld-v0",
#        entry_point="gym_examples.envs:GridWorldEnv",
#        max_episode_steps=300,
#   )

# %%
# The environment ID consists of three components, two of which are
# optional: an optional namespace (here: ``gym_examples``), a mandatory
# name (here: ``GridWorld``) and an optional but recommended version
# (here: v0). It might have also been registered as ``GridWorld-v0`` (the
# recommended approach), ``GridWorld`` or ``gym_examples/GridWorld``, and
# the appropriate ID should then be used during environment creation.
#
# The keyword argument ``max_episode_steps=300`` will ensure that
# GridWorld environments that are instantiated via ``gymnasium.make`` will
# be wrapped in a ``TimeLimit`` wrapper (see `the wrapper
# documentation </api/wrappers>`__ for more information). A done signal
# will then be produced if the agent has reached the target *or* 300 steps
# have been executed in the current episode. To distinguish truncation and
# termination, you can check ``info["TimeLimit.truncated"]``.
#
# Apart from ``id`` and ``entrypoint``, you may pass the following
# additional keyword arguments to ``register``:
#
# +----------------------+-----------+-----------+---------------------------------------------------------------------------------------------------------------+
# | Name                 | Type      | Default   | Description                                                                                                   |
# +======================+===========+===========+===============================================================================================================+
# | ``reward_threshold`` | ``float`` | ``None``  | The reward threshold before the task is  considered solved                                                    |
# +----------------------+-----------+-----------+---------------------------------------------------------------------------------------------------------------+
# | ``nondeterministic`` | ``bool``  | ``False`` | Whether this environment is non-deterministic even after seeding                                              |
# +----------------------+-----------+-----------+---------------------------------------------------------------------------------------------------------------+
# | ``max_episode_steps``| ``int``   | ``None``  | The maximum number of steps that an episode can consist of. If not ``None``, a ``TimeLimit`` wrapper is added |
# +----------------------+-----------+-----------+---------------------------------------------------------------------------------------------------------------+
# | ``order_enforce``    | ``bool``  | ``True``  | Whether to wrap the environment in an  ``OrderEnforcing`` wrapper                                             |
# +----------------------+-----------+-----------+---------------------------------------------------------------------------------------------------------------+
# | ``autoreset``        | ``bool``  | ``False`` | Whether to wrap the environment in an ``AutoResetWrapper``                                                    |
# +----------------------+-----------+-----------+---------------------------------------------------------------------------------------------------------------+
# | ``kwargs``           | ``dict``  | ``{}``    | The default kwargs to pass to the environment class                                                           |
# +----------------------+-----------+-----------+---------------------------------------------------------------------------------------------------------------+
#
# Most of these keywords (except for ``max_episode_steps``,
# ``order_enforce`` and ``kwargs``) do not alter the behavior of
# environment instances but merely provide some extra information about
# your environment. After registration, our custom ``GridWorldEnv``
# environment can be created with
# ``env = gymnasium.make('gym_examples/GridWorld-v0')``.
#
# ``gym-examples/gym_examples/envs/__init__.py`` should have:
#
# .. code:: python
#
#    from gym_examples.envs.grid_world import GridWorldEnv
#
# If your environment is not registered, you may optionally pass a module
# to import, that would register your environment before creating it like
# this - ``env = gymnasium.make('module:Env-v0')``, where ``module``
# contains the registration code. For the GridWorld env, the registration
# code is run by importing ``gym_examples`` so if it were not possible to
# import gym_examples explicitly, you could register while making by
# ``env = gymnasium.make('gym_examples:gym_examples/GridWorld-v0)``. This
# is especially useful when you’re allowed to pass only the environment ID
# into a third-party codebase (eg. learning library). This lets you
# register your environment without needing to edit the library’s source
# code.

# %%
# Creating a Package
# ------------------
#
# The last step is to structure our code as a Python package. This
# involves configuring ``gym-examples/setup.py``. A minimal example of how
# to do so is as follows:
#
# .. code:: python
#
#    from setuptools import setup
#
#    setup(
#        name="gym_examples",
#        version="0.0.1",
#        install_requires=["gymnasium==0.26.0", "pygame==2.1.0"],
#    )
#
# Creating Environment Instances
# ------------------------------
#
# After you have installed your package locally with
# ``pip install -e gym-examples``, you can create an instance of the
# environment via:
#
# .. code:: python
#
#    import gym_examples
#    env = gymnasium.make('gym_examples/GridWorld-v0')
#
# You can also pass keyword arguments of your environment’s constructor to
# ``gymnasium.make`` to customize the environment. In our case, we could
# do:
#
# .. code:: python
#
#    env = gymnasium.make('gym_examples/GridWorld-v0', size=10)
#
# Sometimes, you may find it more convenient to skip registration and call
# the environment’s constructor yourself. Some may find this approach more
# pythonic and environments that are instantiated like this are also
# perfectly fine (but remember to add wrappers as well!).
#
# Using Wrappers
# --------------
#
# Oftentimes, we want to use different variants of a custom environment,
# or we want to modify the behavior of an environment that is provided by
# Gymnasium or some other party. Wrappers allow us to do this without
# changing the environment implementation or adding any boilerplate code.
# Check out the `wrapper documentation </api/wrappers/>`__ for details on
# how to use wrappers and instructions for implementing your own. In our
# example, observations cannot be used directly in learning code because
# they are dictionaries. However, we don’t actually need to touch our
# environment implementation to fix this! We can simply add a wrapper on
# top of environment instances to flatten observations into a single
# array:
#
# .. code:: python
#
#    import gym_examples
#    from gymnasium.wrappers import FlattenObservation
#
#    env = gymnasium.make('gym_examples/GridWorld-v0')
#    wrapped_env = FlattenObservation(env)
#    print(wrapped_env.reset())     # E.g.  [3 0 3 3], {}
#
# Wrappers have the big advantage that they make environments highly
# modular. For instance, instead of flattening the observations from
# GridWorld, you might only want to look at the relative position of the
# target and the agent. In the section on
# `ObservationWrappers </api/wrappers/#observationwrapper>`__ we have
# implemented a wrapper that does this job. This wrapper is also available
# in gym-examples:
#
# .. code:: python
#
#    import gym_examples
#    from gym_examples.wrappers import RelativePosition
#
#    env = gymnasium.make('gym_examples/GridWorld-v0')
#    wrapped_env = RelativePosition(env)
#    print(wrapped_env.reset())     # E.g.  [-3  3], {}
