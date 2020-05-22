from MCRL.agents import Agent
from MCRL.mouselab import MouselabEnv
from MCRL.distributions import Categorical, Normal
from MCRL.modified_mouselab_new import TrialSequence
from MCRL.policies import FunctionPolicy
from MCRL.exact import solve
from contexttimer import Timer
from frozendict import frozendict
from hyperparams import BRANCHING, STRATEGY, STRATEGY_NAME, ENV_TYPE, DISTRIBUTION

import gym
import os
import numpy as np
import pickle
import copy
import time
import collections
import math
import argparse

from pympler.tracker import SummaryTracker
tracker = SummaryTracker()
import gc



def reward(depth):
    """
    Return the Categorical distribution for values of nodes on level 'depth'.

    Parameters
    ----------
    depth : int

    Returns
    -------
    distributions.Categorical
    """
    if depth > 0:
        return Categorical(DISTRIBUTION[depth])
    return 0 

def optimal_policy(env, return_qvalue=False, verbose=False):
    """
    Compute the optimal policy for environment 'env'.

    Parameters
    ----------
    env : Trial
    return_qvalue : bool
    verbose : bool

    Returns
    -------
    policy : FunctionPolicy
    Q : function( [ Categorical/Normal/int ], int )
    """
    with Timer() as t:
        Q, V, pi, info = solve(env)
        v = V(env.init)
    if verbose:
        print('optimal -> {:.2f} in {:.3f} sec'.format(v, t.elapsed))

    policy = FunctionPolicy(pi)
    if return_qvalue:
        return policy, Q
    else:
        return policy

def make_env(cost=1, branching=BRANCHING, distr=DISTRIBUTION, ground_truth=None, 
             seed=None):
    """
    Create an instance of the class MouselabEnv.

    Parameters
    ----------
    cost : int
        The cost of clicking
    branching : [ int ]
        The number of nodes on each level
    distr : dict
        int : [ int ]
            The support of the categorical distribution of the Mouselab MDP's 
            level defined by the key of the dictionary. Allows to train the 
            policy on one type of an environment but create demonstrations for 
            another
    ground_truth : [ int ]
        Values for the nodes. Nodes are numbered along the branches.
    seed : int

    Returns
    -------
    mouselab_env : MouselabEnv
    """
    if seed is not None:
        np.random.seed(seed)

    def reward(depth):
        if depth > 0:
            return Categorical(distr[depth])
        return 0 

    gt = None
    if ground_truth is not None:
        gt = ground_truth
        
    mouselab_env = MouselabEnv.new_symmetric(branching, 
                                             reward, 
                                             cost=cost, 
                                             sample_term_reward=False,
                                             ground_truth=gt)

    return mouselab_env

def make_modified_env(branching=BRANCHING, ground_truth=None, seed=None):
    """
    Create an instance of the class Trial implementing the Mouselab MDP.

    Parameters
    ----------
    branching : [ int ]
        The number of nodes on each level of the Mouselab MDP
    variances : dict
        int : [ int ]
            The support of the categorical distribution of the Mouselab MDP's 
            level defined by the key of the dictionary
    ground_truth : [ int ]
        Values for the nodes. Nodes are numbered along the branches.
    seed : int

    Returns
    -------
    Trial
    """
    if seed is not None:
        np.random.seed(seed)

    if ground_truth is not None:
        ground_truth = [ground_truth]

    seq = TrialSequence(num_trials = 1, 
                        pipeline = [(branching, reward)], 
                        ground_truth = ground_truth)

    return seq.trial_sequence[0]

def get_modified_demo(env, strategy, policy, q_func, 
                      just_actions=False, reward=False):
    """
    Generate one demonstration of the strategy 'strategy' on environment 'env'.

    Save (visited states, taken actions) pairs and (states, non-optimal actions) 
    pairs.

    Parameters
    ----------
    env : Trial
    strategy : function
        A function that turns the generated actions into (state, action) pairs 
        where state is a Trial class modified by the previous action (if exists)
    policy : FunctionPolicy
    q_func : function( [ Categorical/Normal/int ], int )
        The Q function for FunctionPolicy policy that as input accepts 
        (state, action) pairs.
    just_actions : bool
    reward : bool

    Returns
    -------
    dict
        pos : [ (Trial, int) ]
            The set of generated clicks
        neg : [ (Trial, int) ]
            The set of clicks for which the Q function was not the highest
    actions (optional) : [ int ]
        Actions taken during the demonstration
    reward_(optional) : float
        The reward obtained in the demonstration
    """
    env.reset_observations()

    env2 = make_env(1, BRANCHING, ground_truth=env.ground_truth)
    agent = Agent()
    agent.register(env2)
    agent.register(policy)
    trace = agent.run_episode()

    actions = [action if action != 13 else 0 for action in trace['actions']]
    reward_ = sum(trace['rewards'])

    positive_examples = list(strategy(env, actions))
    res = get_negative_examples(q_func, trace['states'][:-1], positive_examples)
    negative_examples, state_actions_distribution = res[0], res[1]

    if just_actions:
        if reward:
            return actions, state_actions_distribution, reward_
        else:
            return actions, state_actions_distribution

    if reward:
        return {'pos': positive_examples, 'neg': negative_examples}, reward_
    else:
        return {'pos': positive_examples, 'neg': negative_examples}


def get_negative_examples(q_func, states, copycat_run):
    """
    Generate (state, action) pairs for actions which are suboptimal under the Q
    function 'q_func'.

    Parameters
    ----------
    q_func : function( [ Categorical/Normal/int ], int )
    states : [ [ Categorical/Normal/int ] ]
    copycat_run : [ (Trial, int) ]
        (State, action) pairs gathered during a demonstration

    Returns
    -------
    negative_samples : [ (Trial, int) ]
    state_actions_distribution : dict
        frozendict([ int ] : [ int ]): [ int ]
            The set of admissable clicks for an observed state written as 
            [node_id] : [value]
    """
    negative_examples = []
    state_actions_distribution = {}
    trial_states = copycat_run

    counter = -1
    for state in states:
        counter += 1
        q_list = {}
        
        t_state = trial_states[counter][0]
        unobserved = [n.label for n in t_state.unobserved_nodes \
                      if n.label != 0]
        observed = [n.label for n in t_state.observed_nodes \
                    if n.label != 0]
        observed_vals = [n.value for n in t_state.observed_nodes \
                         if n.label != 0]

        for action in unobserved + [len(state)]:
            q_list[action] = (q_func(state, action))

        max_q = max(q_list.values())
        q_list = {key: q / max_q if max_q != 0 else q \
                  for key, q in q_list.items()}
        admissable_actions = []

        for action in unobserved + [len(state)]:
            
            if (q_list[action] != 1 and max_q != 0) or \
               (q_list[action] < 0):
                   
                act = action if action != 13 else 0
                negative_examples.append((t_state, act))
                
            if (q_list[action] == 1 and max_q != 0) or \
               (q_list[action] == 0 and max_q == 0):
                   
                act = action if action != 13 else 0
                admissable_actions.append(act)

        for action in observed:
                negative_examples.append((t_state, action))

        froze_state = frozendict({k: v for k,v in zip(observed, observed_vals)})
        froze_action = frozenset(admissable_actions)
        state_actions_distribution[froze_state] = froze_action

    return negative_examples, state_actions_distribution

def get_modified_demonstrations(demo_numbers, env_branching, strategy, policy, 
                                q_func):
    """
    Generate positive and negative (see get_negative_examples) demonstrations
    for the specified environment, following the input RL policy. 

    See 'get_modified_demo' for parameters description.

    Parameters
    ----------
    demo_numbers : range(int)
    env_branching : [ int ]
    strategy : function
    policy : FunctionPolicy
    q_func : function( [ Categorical/Normal/int ], int )

    Returns
    -------
    demonstrations : dict
        pos: [ (Trial, action) ]
            Demonstrated (state, action) pairs
        neg: [ (Trial, action) ]
            (state, action) pairs for encountered states and non-optimal actions 
            according to the RL policy
        leng_pos: [ int ]
            Lengths of the consecutive demonstrations
        leng_neg: [ int ]
            Lengths of the negative examples for the consecutive demonstrations
    further_demonstrations : dict
        A dictionary of the same type as 'demonstrations' used for validating 
        the quality of the program found by the method from 
        decision_tree_imitation_learning
    """
    count = 0
    empty_demo_dict = {'pos': [], 'neg': [], 'leng_pos': [], 'leng_neg': []}
    demonstrations = empty_demo_dict
    further_demonstrations = empty_demo_dict

    for i in demo_numbers:
        env = make_modified_env(env_branching)
        demo = get_modified_demo(env, strategy, policy, q_func)
        f_env = make_modified_env(env_branching)
        f_demo = get_modified_demo(f_env, strategy, policy, q_func)

        demonstrations['pos'].extend(demo['pos'])
        demonstrations['neg'].extend(demo['neg'])
        demonstrations['leng_pos'].append(len(demo['pos']))
        demonstrations['leng_neg'].append(len(demo['neg']))
        further_demonstrations['pos'].extend(f_demo['pos'])
        further_demonstrations['neg'].extend(f_demo['neg'])
        further_demonstrations['leng_pos'].append(len(f_demo['pos']))
        further_demonstrations['leng_neg'].append(len(f_demo['neg']))
        print(demonstrations)

    return demonstrations, further_demonstrations

def get_test_stats(demo_numbers, env_branching, strategy, policy, q_func):
    """
    Generate data needed to measure the quality of PLP interpretations found
    by the 'interpret_binary' or 'interpret_adaptive' algorithms.

    See 'get_modified_demo' for parameters description.

    Parameters
    ----------
    demo_numbers : range(int)
    env_branching : [ int ]
    strategy : function
    policy : FunctionPolicy
    q_func : function( [ Categorical/Normal/int ], int )

    Returns
    -------
    test_moves_all : [ [ int ] ]
        The list of all the action sequences taken by the expert
    mean_test_reward : [ int ]
        The list of rewards averaged over 10 000 trials in each generated 
        environment
    test_envs : [ Trial ]
        The list of generated environments
    state_actions_distr : dict
        int : frozendict([ int ] : [ int ]): [ int ]
            The set of admissable clicks for observed states written as 
            [node_id] : [value] keyed with a number corresponding to a
            test environment
    """
    test_moves_all = tuple()
    test_envs = []
    mean_rewards_sum = []
    state_actions_distr = {i: {} for i in demo_numbers}
    total = 0

    for i in demo_numbers:
        tracker.print_diff()
        
        env = make_modified_env(env_branching)
        test_envs.append(copy.deepcopy(env))
        
        demos = [get_modified_demo(env, 
                                   strategy,
                                   policy=policy, 
                                   q_func=q_func, 
                                   just_actions=True, 
                                   reward=True) for _ in range(10000)]
        
        rewards = [d[-1] for d in demos]
        test_moves = tuple(tuple(d[0]) for d in demos)
        test_moves_all += test_moves
        mean_rewards_sum.append(sum(rewards) / 10000)

        for dct in [d[1] for d in demos]:
            #make state_(...)_ution a dict keyed by tried environments with 
            #values as dicts of encountered states: action distribution
            state_actions_distr[i] = {**state_actions_distr[i], **dct}
            
        total += len(state_actions_distr[i])
        print("#state observed after iteration {}: {}\n".format(i, total))

        del env #?
        del demos #?
        del rewards #?
        del test_moves #?
        gc.collect()

    mean_test_reward = mean_rewards_sum
    
    return [test_moves_all, mean_test_reward], test_envs, state_actions_distr

def get_demos_and_stats(num_demos, env_branching, strategy, policy, q_func):
    """
    Parameters
    ----------
    num_demos : int
    env_branching : [ int ]
    strategy : function
    policy : FunctionPolicy
    q_func : function( [ Categorical/Normal/int ], int )

    Returns
    -------
    demonstrations : dict
    further_demonstrations : dict
    test_clicks : ( [ [ int ] ], [ int ] )
        A tuple of 'test_moves_all' and 'mean_test_reward' from get_test_stats
    test_envs : [ Trial ]
    state_actions_distr : dict
    """
    demos, further_demos = get_modified_demonstrations(range(num_demos),
                                                       BRANCHING, 
                                                       strategy, 
                                                       policy, 
                                                       q_func)
    
    res = get_test_stats(range(num_demos), BRANCHING, strategy, policy, q_func)
    test_clicks, test_envs, state_action_distr = res[0], res[1], res[2]
    
    return demos, further_demos, test_clicks, test_envs, state_action_distr

def save_demos_and_stats(num_demos, demonstrations, further_demonstrations, 
                         test_clicks, test_envs, state_actions_distribution, n):
    """
    A function for saving the generated data in the cwd/PLP_data folder.

    Parameters
    ----------
    num_demos : int
    demonstrations : dict
    further_demonstrations : dict
    test_clicks : ( [ [ int ] ], [ int ] )
    test_envs : [ Trial ]
    state_actions_distribution : dict
    """
    cwd = os.getcwd()
    folder_path = cwd + "/PLP_data/"
   
    ENV = '_' + ENV_TYPE if ENV_TYPE != 'standard' else ''
    NUM_DEMO = "_" + str(num_demos)
    C = "10000_"
    N = str(n) + ".pkl" if n != 0 else ".pkl"
    
    demos_file = STRATEGY_NAME + NUM_DEMO + ENV + N
    demos_path = folder_path + demos_file
    
    with open(demos_path, "wb") as filename:
        pickle.dump(demonstrations, filename)

    clicks_file = "test_actions_" + STRATEGY_NAME + NUM_DEMO + C + ENV + N
    clicks_path = folder_path + clicks_file
    
    with open(clicks_path, "wb") as filename2:
        pickle.dump(test_clicks, filename2)
    
    envs_file = "test_envs_" + STRATEGY_NAME + NUM_DEMO + ENV + N
    envs_path = folder_path + envs_file
    
    with open(envs_path, "wb") as filename3:
        pickle.dump(test_envs, filename3)
        
    f_demos_file = "further_" + STRATEGY_NAME + NUM_DEMO + ENV + N
    f_demos_path = folder_path + f_demos_file
    
    with open(f_demos_path, "wb") as filename4:
        pickle.dump(further_demonstrations, filename4)
        
    sad_file = "distribution_" + STRATEGY_NAME + NUM_DEMO + ENV + N
    sad_path = folder_path + sad_file

    with open(sad_path, "wb") as filename5:
        pickle.dump(state_actions_distribution, filename5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', '-n', type=int, help='', default=1)

    args = parser.parse_args()

    e2 = make_env(1, BRANCHING, DISTRIBUTION)
    strategy = STRATEGY

    print("Computing the optimal policy...")
    opt_policy, Q = optimal_policy(e2, return_qvalue=True, verbose=True)

    import sys
    sys.exit()

    print("Getting the demonstrations and their statistics...")
    st = time.time()
    for i in range(args.num):

        res = get_demos_and_stats(num_demos=64,
                                  env_branching=BRANCHING,
                                  strategy=strategy, 
                                  policy=opt_policy, 
                                  q_func=Q)

        demos, val_demos = res[0], res[1]
        test_clicks, test_envs = res[2], res[3]
        state_actions_distr = res[4]

        print("Saving those bad boys...")
        save_demos_and_stats(64, demos, 
                             val_demos, 
                             test_clicks, 
                             test_envs, 
                             state_actions_distr,
                             n=i)
    end = time.time()

    print("Passed: {}".format(end-st))
