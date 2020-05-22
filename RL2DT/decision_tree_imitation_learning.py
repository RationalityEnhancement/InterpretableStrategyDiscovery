from PLP.DSL import *
from PLP.grammar_utils import generate_programs
from PLP.dt_utils import extract_plp_from_dt
from PLP.policy import StateActionProgram, PLPPolicy
from strategy_demonstrations import \
                             get_modified_demo as get_demo, \
                             get_modified_demonstrations as get_demonstrations,\
                             make_modified_env as make_env
from mouselab_utils import env_to_string, compare, solve_mouselab
from hyperparams import STRATEGY_NAME

from functools import partial
from sklearn.tree import DecisionTreeClassifier
from scipy.special import logsumexp
from scipy.sparse import csr_matrix, lil_matrix, vstack

import multiprocessing
import numpy as np
import pandas as pd
import time
import os
import math
import pickle
import copy

"""
Copyright ©:
Silver et al. (2019). Few-Shot Bayesian Imitation Learning with Logic over Programs.

Changed slightly to accomodate for Mouselab and the intepretation algorithm.
"""

def get_program_set(num_programs, verbose=True):
    """
    Enumerate all programs up to a certain iteration.

    Parameters
    ----------
    num_programs : int

    Returns
    -------
    programs : [ StateActionProgram ]
        A list of programs in enumeration order.
    program_prior_log_probs : [ float ]
        Log probabilities for each program.
    """
    grammar = create_grammar()

    program_generator = generate_programs(grammar)
    programs = []
    program_prior_log_probs = []

    if verbose: print("Generating {} programs".format(num_programs))
    for _ in range(num_programs):
        program, lp = next(program_generator)
        programs.append(program)
        program_prior_log_probs.append(lp)
    if verbose: print("\nDone.")

    return programs, program_prior_log_probs

def apply_programs(programs, fn_input):
    """
    Worker function that applies a list of programs to a single given input.

    Parameters
    ----------
    programs : [ callable ]
    fn_input : Any

    Returns
    -------
    results : [ bool ]
        Program outputs in order.
    """
    count = 0
    x = []
    for program in programs:
        x_i = program(*fn_input)
        x.append(x_i)
    return x

def run_all_programs_on_demonstrations(num_programs, input_demonstrations, 
                                       program_interval=1000, verbose=True):
    """
    Run all programs up to some iteration on one demonstration.

    Expensive in general because programs can be slow and numerous, so caching 
    can be very helpful.

    Parallelization is designed to save time in the regime of many programs.

    Care is taken to avoid memory issues, which are a serious problem when 
    num_programs exceeds 50,000.

    Returns classification dataset X, y.

    Parameters
    ----------
    num_programs : int
    input_demonstrations : dict 
        pos: [ (Trial, action) ]
            Demonstrated (state, action) pairs
        neg: [ (Trial, action) ]
            (state, action) pairs for encountered states and non-optimal actions 
            according to the RL policy
        leng_pos: [ int ]
            Lengths of the consecutive demonstrations
        leng_neg: [ int ]
            Lengths of the negative examples for the consecutive demonstrations
    program_interval : int
        This interval splits up program batches for parallelization.

    Returns
    -------
    X : csr_matrix
        X.shape = (num_demo_items, num_programs)
    y : [ int(bool) ]
        y.shape = (num_demo_items,)
    """

    if verbose: print("Running all programs on the Mouselab MDP and input demonstrations")

    programs, _ = get_program_set(num_programs)

    assert type(input_demonstrations) is dict and \
           set(input_demonstrations.keys()).issuperset(set(['pos', 'neg'])), \
           "Input demonstrations for the copycat policy are not a dict or do not \
            have positive and negative examples"

    positive_examples = input_demonstrations['pos']
    negative_examples = input_demonstrations['neg']

    y = [1] * len(positive_examples) + [0] * len(negative_examples)

    num_data = len(y)
    num_programs = len(programs)

    X = lil_matrix((num_data, num_programs), dtype=bool)

    # This loop avoids memory issues
    for i in range(0, num_programs, program_interval):
        end = min(i+program_interval, num_programs)
        if verbose: print('Iteration {} of {}'.format(i, num_programs), end='\r')
        num_workers = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(num_workers)

        fn = partial(apply_programs, programs[i:end])
        fn_inputs = positive_examples + negative_examples
        
        results = pool.map(fn, fn_inputs)
        pool.close()

        for X_idx, x in enumerate(results):
            X[X_idx, i:end] = x

    X = X.tocsr()
    y = np.array(y, dtype=np.uint8)

    print()
    return X, y

def learn_single_batch_decision_trees(y, num_dts, X_i, tree_depth):
    """
    Parameters
    ----------
    y : [ bool ]
    num_dts : int
        The number of trees to generate
    X_i : csr_matrix
    tree_depth : int
        The maximum depth of the trees

    Returns
    -------
    clfs : [ DecisionTreeClassifier ]
    """
    clfs = []

    for seed in range(num_dts):
        clf = DecisionTreeClassifier(criterion = "entropy", 
                                     min_samples_split = 2, 
                                     max_depth = tree_depth, 
                                     random_state=seed)
        clf.fit(X_i, y)
        clfs.append(clf)

    return clfs

def learn_plps(X, y, programs, program_prior_log_probs, num_dts=1, 
               program_generation_step_size=10, tree_depth=6,
               verbose=True):
    """
    Parameters
    ----------
    X : csr_matrix
    y : [ bool ]
    programs : [ StateActionProgram ]
    program_prior_log_probs : [ float ]
    num_dts : int
    program_generation_step_size : int
    tree_depth : int

    Returns
    -------
    plps : [ StateActionProgram ]
    plp_priors : [ float ]
        Log probabilities.
    """
    plps = []
    plp_priors = []
    prog_gen_st_si = program_generation_step_size

    num_programs = len(programs)

    #for i in range(0, num_programs, program_generation_step_size):
    i = 0
    while i < num_programs:
        pgss = prog_gen_st_si if i+prog_gen_st_si <= num_programs \
               else num_programs-i
        i += pgss
        if verbose: print("Learning plps with {} programs".format(i))
        for clf in learn_single_batch_decision_trees(y, num_dts, X[:, :i+1], tree_depth):
            plp, plp_prior_log_prob = extract_plp_from_dt(clf, programs, 
                                                          program_prior_log_probs) 
            plps.append(plp)
            plp_priors.append(plp_prior_log_prob)

    return plps, plp_priors

def compute_likelihood_single_plp(demonstrations, plp):
    """
    Parameters
    ----------
    demonstrations : [ (Trial, int) ]
        State, action pairs.
    plp : StateActionProgram
    
    Returns
    -------
    likelihood : float
        The log likelihood.
    """
    ll = 0.

    for obs, action in demonstrations:

        ### ad-hoc for Mouselab ###
        #### removing stopping ####
        if action == 0:
            pass
        ###########################
        elif not plp(obs, action):
            return -np.inf

            size = 1

            for i in range(obs.num_nodes):
                    if i == action:
                        continue
                    if plp(obs, i):
                        size += 1

            ll += np.log(1. / size)

    return ll

def compute_likelihood_plps(plps, demonstrations):
    """
    See compute_likelihood_single_plp.
    """
    num_workers = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_workers)

    fn = partial(compute_likelihood_single_plp, demonstrations)
    likelihoods = pool.map(fn, plps)
    pool.close()

    return likelihoods

def select_particles(particles, particle_log_probs, max_num_particles):
    """
    Parameters
    ----------
    particles : [ Any ]
    particle_log_probs : [ float ]
    max_num_particles : int

    Returns
    -------
    selected_particles : [ Any ]
    selected_particle_log_probs : [ float ]
    """
    sorted_log_probs, _, sorted_particles = (list(t) \
        for t in zip(*sorted(zip(particle_log_probs, 
                                 np.random.random(size=len(particles)), particles), reverse=True)))
    end = min(max_num_particles, len(sorted_particles))
    try:
        idx = sorted_log_probs.index(-np.inf)
        end = min(idx, end)
    except ValueError:
        pass
    return sorted_particles[:end], sorted_log_probs[:end]


def train(program_gen_step_size, num_programs, num_dts, max_num_particles, 
          input_demos, further_demos, tree_depth=6, return_prior=False, 
          pred_data=[None, None], verbose=True):
    """
    Parameters
    ----------
    program_gen_step_size : int
    num_programs : int
    num_dts : int
        The number of decision trees to generate before adding a new predicate
    max_num_particles : int
    input_demos : dict
        pos: [ (Trial, action) ]
            Demonstrated (state, action) pairs
        neg: [ (Trial, action) ]
            (state, action) pairs for encountered states and non-optimal actions
            according to the RL policy
        leng_pos: [ int ]
            Lengths of the consecutive demonstrations
        leng_neg: [ int ]
            Lengths of the negative examples for the consecutive demonstrations
    further_demos : dict
    tree_depth : int
        The maximum depth of the tree used to imitate the demonstrations
    return_prior : bool
    pred_data : [ csr_matrix, int(bool) ]
        Predicates precomputed for the 'input_demos'
    verbose (optional) : bool
    
    Returns
    -------
    StateActionProgram
        The best found program
    float (optional)
        Log prior of the best program
    """

    programs, program_prior_log_probs = get_program_set(num_programs, verbose=verbose)

    input_demonstrations = input_demos
    further_demonstrations = further_demos['pos']

    if pred_data != [None, None]:
        X, y = pred_data[0], pred_data[1]
    else:
        X, y = run_all_programs_on_demonstrations(num_programs, 
                                                  input_demonstrations,
                                                  verbose=verbose)
    plps, plp_priors = learn_plps(X, 
                                  y, 
                                  programs, 
                                  program_prior_log_probs,
                                  num_dts=num_dts, 
                                  program_generation_step_size=program_gen_step_size,
                                  tree_depth=tree_depth,
                                  verbose=verbose)

    likelihoods = compute_likelihood_plps(plps, further_demonstrations)

    particles = []
    particle_log_probs = []

    for plp, prior, likelihood in zip(plps, plp_priors, likelihoods):
        particles.append(plp)
        particle_log_probs.append(prior + likelihood)

    if verbose: print("\nDone!")
    map_idx = np.argmax(particle_log_probs).squeeze()
    if verbose: print("MAP program ({}):".format(np.exp(particle_log_probs[map_idx])))

    top_particles, top_particle_log_probs = select_particles(particles, 
                                                             particle_log_probs, 
                                                             max_num_particles)
    for i in range(len(top_particles)):
        if verbose: 
            print(top_particles[i])
            print("\n")
    if len(top_particle_log_probs) > 0:
        value = np.array(top_particle_log_probs) - logsumexp(top_particle_log_probs)
        top_particle_log_probs = value
        top_particle_probs = np.exp(top_particle_log_probs)
        if verbose: print("top_particle_probs:", top_particle_probs)
        policy = PLPPolicy(top_particles, top_particle_probs)
    else:
        if verbose: print("no nontrivial particles found")
        policy = PLPPolicy([StateActionProgram("False")], [1.0])

    if len(top_particles) != 0:
        if not return_prior:
            return top_particles[0]
        else:
            interpretable_tree_posterior = particle_log_probs[map_idx]
            return top_particles[0], np.exp(interpretable_tree_posterior)
    if not return_prior:
        return None
    else:    
        return None, 0

def formula_validation(best_formula, expert_reward, num_rollouts, 
                       rollout_function=solve_mouselab, stats=None):
    """
    Compute the input PLP's mean return and print a number of statistics

    Parameters
    ----------
    best_formula : StateActionProgram
        The PLP to be tested against the expert's strategy
    expert_reward : float
        The mean reward obtained by the expert after num_rollouts
    num_rollouts : int
        Number of times to initialize a new state and run the policy induced from
        the best_formula until termination
    rollout_function (optional) : function
        Function that performs a rollout of the policy induced by the formula in
        the environment that this policy regards
    stats (optional) : tuple
        See compute_agreement_statistics
        
    Returns
    -------
    rew_expert : int
        The mean reward of the expert gained when acting in environments from 
        test_envs
    rew_formula
        The mean reward of best_formula gained when acting in environments from 
        test_envs
    """
    
    formula_rewards = []

    for i in range(num_rollouts):
        print("\rTesting env {}".format(i+1), end='')

        env = make_env()

        rollout = rollout_function(best_formula, env)
        formula_rewards.append(rollout[1])

    rew_expert = expert_reward
    rew_formula = sum(formula_rewards) / num_rollouts
    
    if stats != None: 
        compute_agreement_statistics(best_formula, stats[0], stats[1], stats[-1])
    
    print("\n\nExpert received an average reward of {}, ".format(rew_expert) + \
          "whereas the formula received an average reward of {} ".format(rew_formula) + \
          "which makes for {}% of the optimal".format(rew_formula/rew_expert * 100))

    return rew_expert, rew_formula

def compute_agreement_statistics(formula, valid_envs, valid_actions, 
                                 valid_distribution):
    '''
    Print a number of statistics regarding the formula-induced policy with
    respect to the expert in a set of precomputed environments

    Parameters
    ----------
    formula : StateActionProgram
        The PLP to be tested against the expert's strategy
    expert_reward : float
        The mean reward obtained by the expert after num_rollouts
    valid_envs : [ Any ]
        A list of environments on which to evaluate the PLP and on which the 
        expert has been already evaluated (when gathering the data)
    valid_actions : ( ( int ) )
        A tuple of tuples encoding actions taken by the expert in each
        environment from test_envs
    valid_distribution : dict
        frozendict([ int ] : [ int ]): [ int ]
            A et of admissable actions for observed states written as 
            [node_id] : [value] encountered when the expert acted in 
            environments from test_envs
    '''
    num_envs = len(valid_envs)
    actions_all_formula = tuple()
    actions_all_expert = valid_actions
    state_actions_distr = {i: {} for i in range(num_envs)}
    list_of_envs = []

    st = time.time()
    for i in range(num_envs):
        valid_env = copy.deepcopy(valid_envs[i])
        list_of_envs.append(env_to_string(valid_env))
        valid_rollout = solve_mouselab(formula, valid_env)
        actions_env_formula = tuple(tuple(valid_rollout[0]))
        state_actions_distr[i] = {**state_actions_distr[i], **valid_rollout[2]}
        actions_all_formula += actions_env_formula
    end = time.time()
    print("Computing the statistics took {} s".format(end-st))
    
    res = compare(state_actions_distr, test_distribution)
    distributions_fit = res[0]
    broader_distribution, slimmer_distribution = res[1], res[2]

    actions_per_env_formula = {str(env_str): [] for env_str in list_of_envs}
    actions_per_env_expert = {str(env_str): [] for env_str in list_of_envs}

    for i in range(num_rollouts):
        env_str = str(list_of_envs[num_rollouts])
        if actions_all_formula[i] not in actions_per_env_formula[env_str]:
            actions_per_env_formula[env_str].append(actions_all_formula[i])
                
        if actions_all_expert[i] not in actions_per_env_expert[env_str]:
            actions_per_env_expert[env_str].append(actions_all_expert[i])

    broader_actions_expert = {}
    actions_all_formula = tuple()
    actions_all_expert = valid_actions
    state_actions_distr = {i: {} for i in range(num_envs)}
    list_of_envs = []
    broader_actions_formula = {}
    for k in actions_per_env_formula.keys():
            
        set_actions_env_k_formula = set(actions_per_env_formula[k])
        set_actions_env_k_expert = set(actions_per_env_expert[k])
            
        broader_actions_expert[k] = set_actions_env_k_expert-set_actions_env_k_formula
        broader_actions_formula[k] = set_actions_env_k_formula-set_actions_env_k_expert

    sub = [set(actions_per_env_formula[i]).issubset(set(actions_per_env_expert[i]))
           for i in actions_per_env_formula.keys()]
    included = all(sub)
        
    sup = [set(actions_per_env_formula[i]).issuperset(set(actions_per_env_expert[i]))
           for i in actions_per_env_formula.keys()]
    superset = all(sup)

    equal = included and superset
        
    def IOU(list1, list2):
        intersection = set(list1).intersection(set(list2))
        intersection_size = len(intersection)
        union = set(list1).union(set(list2))
        union_size = len(union)
        return float(intersection_size) / union_size
        
    agreement = [IOU(actions_per_env_formula[i], actions_per_env_expert[i])
                 for i in actions_per_env_formula.keys()]
    mean_agreement = np.mean(agreement)

    print("The FORMULA is equal to the expert: {}, \n".format(equal) + \
          "More conservative: {}, \n".format(included) + \
          "Broader: {}. \n".format(superset) + \
          "The FORMULA agrees with the EXPERT in {} %".format(100*mean_agreement))

    print(np.array(distributions_fit))



if __name__  == "__main__":

    cwd = os.getcwd()
    path_demos = cwd+"/PLP_data/" + STRATEGY_NAME + "_256.pkl"
    path_demos_further = cwd+"/PLP_data/further_" + STRATEGY_NAME + "_256.pkl"
    path_envs =  cwd+"/PLP_data/test_envs_" + STRATEGY_NAME + "_256.pkl"
    path_actions =  cwd+"/PLP_data/test_actions_" + STRATEGY_NAME + "_256_10000.pkl"
    path_distribution = cwd+"/PLP_data/distribution_" + STRATEGY_NAME + "_256.pkl"

    with open(path_demos, 'rb') as handle:
        demos = pickle.load(handle)

    with open(path_demos_further, 'rb') as handle:
        demos_further = pickle.load(handle)

    with open(path_envs, 'rb') as handle:
        envs = pickle.load(handle)

    with open(path_actions, 'rb') as handle:
        actions, mean_reward = pickle.load(handle)

    with open(path_distribution, 'rb') as handle:
        test_distribution = pickle.load(handle)

    best_tree = train(1, 100, 5, 25, demos[:2], demos_further[:2])

    tree_test(best_tree, envs[:2], actions[:2], mean_reward, choose(test_distribution, 1))
