import time
import os
from dataclasses import dataclass, field
import ioh

import numpy as np
import gym

from network import Layer, argmax, MinMaxNormalizer

import sys

sys.path.append("..")

from algorithms.csa import CSA
from algorithms.maes import MAES
from algorithms.cmaes import CMAES


@dataclass
class Policy:
    n_agents: int
    lb: np.array
    ub: np.array
    n_actions: int
    state_size: int = 0
    m_agent: int = 4
    add_agent_layers: bool = True  
    n_weights: int = 0
    normalizer: MinMaxNormalizer = None
    w: np.ndarray = None

    def __post_init__(self):
        self.state_size = len(self.ub)
        self.normalizer = MinMaxNormalizer(self.lb, self.ub)
        if self.add_agent_layers:
            self.input_layer = Layer(self.state_size, self.m_agent)
            self.agent_layers = [
                Layer(self.m_agent, self.n_actions, argmax) for _ in range(self.n_agents)
            ]
            self.n_weights = self.state_size * self.m_agent * self.n_actions * self.n_agents
        else:
            self.input_layer = Layer(self.state_size, self.n_actions, argmax)
            self.agent_layers = []
            self.n_weights = self.state_size * self.n_actions

        if self.w is None:
            self.w = np.zeros(self.n_weights)

        self.set_weight_views()

    def set_weight_views(self):
        idx = 0
        for layer in [self.input_layer] + self.agent_layers:
            layer.set_weights(self.w, idx)
            idx += layer.size
            

    def set_weights(self, w):
        self.w = w.copy()
        self.set_weight_views()
        
    def select_actions(self, observations):
        # observations = self.normalizer(observations)
        activation = self.input_layer(observations)
        if not self.add_agent_layers:
            return activation
        return [
            layer(act.reshape(1, -1))[0]
            for act, layer in zip(activation, self.agent_layers)
        ]


def compress_observation(obs):
    x, y, *grid = obs
    grid = np.array(grid).reshape(3, 3, 5)
    grid = np.c_[np.zeros((3, 3, 1)), grid].argmax(axis=2) / 6
    obs = np.r_[x, y, grid.ravel()]
    return obs


@dataclass
class ObjectiveFunction:
    env_name: str
    n_train_timesteps: int = 0
    n_train_episodes: int = 0
    dimension: int = 0
    compress: bool = False

    def __post_init__(self):
        self.env = gym.make(self.env_name)
        self.lb = self.env.observation_space[0].low
        self.ub = self.env.observation_space[0].high
        if self.compress:
            self.lb = self.lb[:11]
            self.ub = self.ub[:11]
        
        self.policy = Policy(
            self.env.n_agents,
            # Assuming a single fixed agent type
            self.lb, self.ub,
            self.env.action_space[0].n,
        )
        self.dimension = self.policy.n_weights

    def __call__(self, w, render=False):
        env = gym.make(self.env_name)
        policy = Policy(
            env.n_agents,
            self.lb, 
            self.ub,
            env.action_space[0].n,
        )
        policy.set_weights(w)

        ep_reward = 0
        obs_n = env.reset()
        done_n = [False for _ in range(env.n_agents)]
        while not all(done_n):
            obs_before = obs_n
            if self.compress:
                obs_n = [compress_observation(obs) for obs in obs_n]
                
            actions = policy.select_actions(obs_n)
            obs_n, reward_n, done_n, info = env.step(actions)            
            distance_travelled = np.abs(np.array(obs_before) - np.array(obs_n)).sum()
            
            if distance_travelled == 0:
                ep_reward -= 10
                break
            ep_reward += sum(reward_n)
            self.n_train_timesteps += 1
            if render:
                env.render()
                time.sleep(0.25)
        env.close()
        self.n_train_episodes += 1
        return -ep_reward
    
@dataclass
class ES:
    d: int 
    
    def ask(self, n: int = 1) -> np.ndarray:
        """Returns n solutions"""
        return np.random.uniform(size=(n, self.d))
    
    def tell(self, y: np.ndarray) -> None:
        """Assigns the fitness value to the next y.size solutions"""
        
def evaluate(x: np.ndarray) -> np.ndarray:
    return np.random.uniform(len(x))

# def training_scheme():
#     n_combinations = 2
#     n_agent_iters = 1
#     n_agents = 2
#     agent_lambda = 4
#     happy = False
    
#     shared_layer_es = ES(2)
#     agents_es = [ES(4) for _ in range(n_agents)]
    
#     while not happy:
        
#         for shared_layer in shared_layer_es.ask():
#             shared_layer_fitness = 0
            
#             for k in range(n_agent_iters):
#                 populations = np.array([es.ask(agent_lambda) for es in agents_es])
#                 agent_fitness = np.zeros(agent_lambda)
#                 for _ in range(n_combinations):
#                     perm = np.random.permutation(agent_lambda)
#                     for pi in perm:
#                         x = np.r_[shared_layer, populations[:, pi].ravel()]
#                         agent_fitness[pi] = evaluate(x)
#                         breakpoint()
                    
#                 combinations = [np.random.permutation(agent_lambda) for _ in range(n_combinations)]
#                 combinations = [make_combination(populations) for _ in range(n_combinations)]
#                 combination_fitness = evaluate_combinations(shared_layer, combinations)
#                 assign_fitness(combination_fitness)
#                 shared_layer_fitness = max(shared_layer_fitness, max(combination_fitness))
                
#             shared_layer_es.tell(shared_layer_fitness)
#             break
#         break
    

if __name__ == "__main__":
    np.random.seed(12)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default= "ma_gym:Checkers-v0", type=str)
    args = parser.parse_args()
    obj = ObjectiveFunction(args.env_name)
    print(obj.env_name)
    print(obj.dimension)
    print("n_agents: ", obj.env.n_agents)
    print("n_state: ", obj.env.observation_space[0].shape)
    print("n_actions: ", obj.env.action_space[0].n)
    print("minimal_dimension: ", obj.env.n_agents * obj.env.observation_space[0].shape[0] * obj.env.action_space[0].n)

    function = ioh.wrap_problem(obj, "name", dimension=obj.dimension, lb=-1, ub=1)
    
    from modcma import c_maes
    modules = c_maes.parameters.Modules()
    modules.restart_strategy =  c_maes.options.RestartStrategy.RESTART
    modules.repelling_restart = True
    modules.center_placement = c_maes.options.UNIFORM
    modules.matrix_adaptation = c_maes.options.MatrixAdaptationType.SEPERABLE
    
    settings = c_maes.parameters.Settings(
        obj.dimension, 
        modules, 
        sigma0 = 1/3, 
        budget=1_500_000, 
        verbose=True,
        lb=obj.lb,
        ub=obj.ub,
        target=-83.0,
        lambda0=128,
    )
    parameters = c_maes.Parameters(settings)
    cma = c_maes.ModularCMAES(parameters)
    while not cma.break_conditions():
        try:
            cma.step(function)
            print("Best:", function.state.evaluations, function.state.current_best.y)
        except KeyboardInterrupt:
            break
    
    print()
    print()
    print()
    obj(function.state.current_best.x, True)
    print("Best:", function.state.current_best.y)
    breakpoint()
