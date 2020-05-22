from PLP.DSL import *
from PLP.grammar_utils import generate_programs
from PLP.dt_utils import extract_plp_from_dt
from PLP.policy import StateActionProgram, PLPPolicy
from strategy_demonstrations import \
                             get_modified_demo as get_demo, \
                             get_modified_demonstrations as get_demonstrations,\
                             make_modified_env as make_env
from hyperparams import STRATEGY_NAME

from functools import partial
from sklearn.tree import DecisionTreeClassifier
from scipy.special import logsumexp
from scipy.sparse import csr_matrix, lil_matrix, vstack
from random import shuffle, randint
from frozendict import frozendict

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

def learn_plps(X, y, programs, program_prior_log_probs, num_dts=5, 
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

    num_programs = len(programs)

    for i in range(0, num_programs, program_generation_step_size):
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


def train(program_gen_step_size, num_programs, num_dts, 
          max_num_particles, input_demos, further_demos, 
          tree_depth=6, return_prior=False, pred_data=[None, None],
          verbose=True):
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
    if verbose: print("MAP program ({}):".format(particle_log_probs[map_idx]))

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
            return top_particles[0], np.exp(plp_priors[map_idx])
    if not return_prior:
        return None
    else:    
        return None, 0


def solve_mouselab(dec_tree, env, to_what=None):
    """
    Use the program found by the PLP method to act once in a particular 
    instatiation of the Mouselab MDP.

    Parameters
    ----------
    dec_tree : [ StateActionProgram ]
    env : Trial
    to_what : [ int ]

    Returns
    -------
    clicks : ( int )
        A tuple of taken actions
    reward : int
    state_actions_distribution : dict
        frozendict([ int ] : [ int ]): [ int ]
            A set of admissable clicks for observed states written as [node_id] 
: [value]
    """
    env.reset_observations(to_what=to_what)

    state = env
    actions = [ac for ac in range(state.num_nodes)]
    clicks = tuple()
    state_actions_distribution = {}


    while 1 or len(clicks) < 13:

        if (len(clicks) > 0 and clicks[-1] == 0) or \
            len(clicks) > state.num_nodes: 
                break
            
        all_bad = True
        admissable_actions = []
        shuffle(actions)

        for act in actions:
            if dec_tree(state, act):
                all_bad = False
                admissable_actions.append(act)

        if all_bad: 
            admissable_actions = [0]
            
        choice = randint(0,len(admissable_actions)-1)
        act = admissable_actions[choice]
        clicks = clicks + (act,)
        observed = [n.label for n in state.observed_nodes if n.label != 0]
        observed_vals = [n.value for n in state.observed_nodes if n.label != 0]
        froze_state = frozendict({k: v for k,v in zip(observed, observed_vals)})
        froze_action = frozenset(admissable_actions)
        state_actions_distribution[froze_state] = froze_action
    
        if act not in [n.label for n in state.observed_nodes]: 
            state.node_map[act].observe()


    best_path = [node for node in env.node_map.values() 
                 if is_on_highest_expected_value_path(state, node.label)]
    depths    = {i: [] for i in state.level_map.keys()}

    for node in best_path:
        depths[node.depth].append(node)

    best_path = []
    for dep, nodes in depths.items():
        best_path.append(nodes[0].value)

    reward = sum(best_path) - len(clicks) + 1
    if len(set(clicks)) != len(clicks): 
        reward  = 0 ## penalty for clicking observed nodes; 
                    ## the tree is no good if it does that

    return clicks, reward, state_actions_distribution

def compare(main_dict, compare_dict):
    """
    Generate statistics based on the differences between two distributions of
    actions on a set of common states.

    Parameters
    ----------
    main_dict : dict
        frozendict([ int ] : [ int ]): [ int ]
            A set of admissable clicks for observed states written as 
            [node_id] : [value]
        A dictionary for the PLP program found by the method from Silver et al.
    compare_dict : dict
        A dictionary for the expert strategy generated by 
        'strategy_demonstrations.get_modified_demonstrations'

    Returns
    -------
    percentages : [ int ]
        The percentage of states for which the admissable actions of the PLP 
        are [broader_than, same_as, included_in] the admissable actions of the 
        expert strategy
    output_frame : pd.DataFrame
        elements : ( int )
            A tuple of observed node's ids
 
        size : string 
            The relation of the set of admissable PLP actions to the expert's 
            set actions_main : [ int ]
            Actions allowed by the PLP
            All if size == 'SAME' or 'LESS', only the additional when 
            size == 'MORE'
        actions_compare : [ int ]
            Actions allowed by the expert strategy
            All if size == 'SAME' or 'MORE', only the additional when
            size == 'LESS'
        likelihood_tree : float
            The probability for choosing an admissable action by the PLP
        likelihood_expert : float
            The probability for choosing an admissable action by the expert 
            strategy
    """
    all_main_dict = {}
    all_compare_dict = {}
    for k in main_dict.keys():
        all_main_dict = {**all_main_dict, **main_dict[k]}
    for k in compare_dict.keys():
        all_compare_dict = {**all_compare_dict, **compare_dict[k]}

    compare_keys = all_compare_dict.keys()
    common_keys = [key for key in all_main_dict.keys() if key in compare_keys]
    output_dict = {}

    for key in common_keys:
        
        if all_main_dict[key] == all_compare_dict[key]:
            output_dict[key] = ['same', 
                                all_main_dict[key], 
                                all_compare_dict[key], 
                                1./len(all_main_dict[key]), 
                                1./len(all_main_dict[key])]
            
        elif len(all_main_dict[key]) > len(all_compare_dict[key]):
            output_dict[key] = ['broader',
                                [el for el in all_main_dict[key] 
                                 if el not in all_compare_dict[key]],
                                all_compare_dict[key], 
                                1./len(all_main_dict[key]), 
                                1./len(all_compare_dict[key])]
                                
        else:
            output_dict[key] = ['narrower', 
                                all_main_dict[key], 
                                [el for el in all_compare_dict[key] 
                                 if el not in all_main_dict[key]], 
                                1./len(all_main_dict[key]), 
                                1./len(all_compare_dict[key])]

    all_states = len(output_dict.values())
    
    size = {k: v[0] for k,v in output_dict.items()}
    actions_tree = {k: v[1] for k,v in output_dict.items()}
    actions_expert = {k: v[2] for k,v in output_dict.items()}
    likelihood_tree = {k: v[3] for k,v in output_dict.items()}
    likelihood_expert = {k: v[4] for k,v in output_dict.items()}
    output_frame = pd.DataFrame({'size': size,
                                 'actions_tree': actions_tree,
                                 'actions_expert': actions_expert,
                                 'likelihood_tree': likelihood_tree,
                                 'likelihood_expert': likelihood_expert})

    same_num = output_frame[output_frame['size'] == 'same'].shape[0]
    print("SAME:\n")
    print(same_num)
    distribution_same = output_frame[output_frame['size'] == 'same']
    print(distribution_same)
    print("\n\n")
    
    more_num = output_frame[output_frame['size'] == 'broader'].shape[0]
    print("BROADER:\n")
    print(more_num)
    distribution_broader = output_frame[output_frame['size'] == 'broader']
    print(distribution_broader)
    print("\n\n")
    
    less_num = all_states - same_num - more_num
    print("NARROWER:\n")
    print(less_num)
    distribution_narrower = output_frame[output_frame['size'] == 'narrower']
    print(distribution_narrower)
    print("\n\n")

    percentages = [float(same_num) / all_states, 
                   float(more_num) / all_states, 
                   float(less_num) / all_states]
        
    return percentages, distribution_broader, distribution_narrower

def tree_test(best_tree, test_envs, test_clicks, test_reward, test_distribution):
    """
    Print a number of statistics and compute mean reward obtained by the expert
    and the PLP that is its interpretable version.

    Parameters
    ----------
    best_tree : StateActionProgram
        The PLP to be tested against the expert's strategy
    test_envs : [ Trial ]
        A list of environments on which to evaluate the PLP the expert
        has been already evaluated (when gathering the data)
    test_clicks : ( ( int ) )
        A tuple of tuples encoding actions taken by the expert in each
        environment from test_envs
    test_reward : [ int ]
        The mean reward obtained by the expert when ran on each of the
        environments in test_envs 10 000 times
    test_distribution : dict
        frozendict([ int ] : [ int ]): [ int ]
            A et of admissable clicks for observed states written as 
            [node_id] : [value] encountered when the expert acted in 
            environments from test_envs
    Returns
    -------
    rew_expert : int
        The mean reward the expert harboured when acting in environments from 
        test_envs
    rew_tree
        The mean reward best_tree harboured when acting in environments from 
        test_envs
    """
    clicks_all_tree = tuple()
    clicks_all_expert = test_clicks
    mean_rewards_all_tree = []
    list_of_envs = []
    num_envs = len(test_envs)
    state_actions_distr = {i: {} for i in range(num_envs)}


    for i in range(num_envs):
        print("Testing env {}\n".format(i+1))

        env = copy.deepcopy(test_envs[i])
        print(env.ground_truth)

        list_of_envs.append(env.ground_truth)

        st = time.time()
        tree_runs = [solve_mouselab(best_tree, env) for _ in range(10000)]
        clicks_env_tree = tuple(tuple(r[0]) for r in tree_runs)
        mean_reward_env_tree = sum([r[1] for r in tree_runs]) / len(tree_runs)
        mean_rewards_all_tree.append(mean_reward_env_tree)
        
        for d in [r[2] for r in tree_runs]:
            state_actions_distr[i] = {**state_actions_distr[i], **d}
        end = time.time()
        print("Generating 10 000 tree click sequences + \
               computing actions distribution took {} s".format(end-st))

        clicks_all_tree += clicks_env_tree

    res = compare(state_actions_distr, test_distribution)
    distributions_fit = res[0]
    broader_distribution, slimmer_distribution = res[1], res[2]

    clicks_per_env_tree = {str(env_gt): [] for env_gt in list_of_envs}
    clicks_per_env_expert = {str(env_gt): [] for env_gt in list_of_envs}

    rew_expert = sum(test_reward) / num_envs
    rew_tree = sum(mean_rewards_all_tree) / num_envs


    for i in range(num_envs*10000):
        env_str = str(list_of_envs[math.floor(i / 10000)])
        if clicks_all_tree[i] not in clicks_per_env_tree[env_str]:
            clicks_per_env_tree[env_str].append(clicks_all_tree[i])
            
        if clicks_all_expert[i] not in clicks_per_env_expert[env_str]:
            clicks_per_env_expert[env_str].append(clicks_all_expert[i])

    broader_clicks_expert = {}
    broader_clicks_tree = {}
    for k in clicks_per_env_tree.keys():
        
        set_clicks_env_k_tree = set(clicks_per_env_expert[k])
        set_clicks_env_k_expert = set(clicks_per_env_tree[k])
        
        broader_clicks_expert[k] = set_clicks_env_k_expert-set_clicks_env_k_tree
        broader_clicks_tree[k] = set_clicks_env_k_tree-set_clicks_env_k_expert

    sub = [set(clicks_per_env_tree[i]).issubset(set(clicks_per_env_expert[i]))
           for i in clicks_per_env_tree.keys()]
    included = all(sub)
    
    sup = [set(clicks_per_env_tree[i]).issuperset(set(clicks_per_env_expert[i]))
           for i in clicks_per_env_tree.keys()]
    superset = all(sup)

    equal = included and superset
    
    def IOU(list1, list2):
        intersection = set(list1).intersection(set(list2))
        intersection_size = len(intersection)
        union = set(list1).union(set(list2))
        union_size = len(union)
        return float(intersection_size) / union_size
    
    agreement = [IOU(clicks_per_env_tree[i], clicks_per_env_expert[i])
                 for i in clicks_per_env_tree.keys()]
    mean_agreement = np.mean(agreement)

    print(best_tree)
    print("The TREE is equal to the expert: {}, \n".format(equal) + \
          "More conservative: {}, \n".format(included) + \
          "Broader: {}. \n".format(superset) + \
          "The TREE agrees with the EXPERT in {} %".format(100*mean_agreement))
          
    print("Expert received an average reward of {}, ".format(rew_expert) + \
          "whereas the tree received an average reward of {} ".format(rew_tree) + \
          "which makes for {}% of the optimal".format(rew_tree/rew_expert * 100))

    print(np.array(distributions_fit))
    print(broader_distribution)

    return rew_expert, rew_tree



if __name__  == "__main__":

    cwd = os.getcwd()
    path_demos = cwd+"/PLP_data/" + STRATEGY_NAME + "_256.pkl"
    path_demos_further = cwd+"/PLP_data/further_" + STRATEGY_NAME + "_256.pkl"
    path_envs =  cwd+"/PLP_data/test_envs_" + STRATEGY_NAME + "_256.pkl"
    path_clicks =  cwd+"/PLP_data/test_clicks_" + STRATEGY_NAME + "_256_10000.pkl"
    path_distribution = cwd+"/PLP_data/distribution_" + STRATEGY_NAME + "_256.pkl"

    with open(path_demos, 'rb') as handle:
        demos = pickle.load(handle)

    with open(path_demos_further, 'rb') as handle:
        demos_further = pickle.load(handle)

    with open(path_envs, 'rb') as handle:
        envs = pickle.load(handle)

    with open(path_clicks, 'rb') as handle:
        clicks, mean_reward = pickle.load(handle)

    with open(path_distribution, 'rb') as handle:
        test_distribution = pickle.load(handle)

    best_tree = train(1, 100, 5, 25, demos[:2], demos_further[:2])

    tree_test(best_tree, envs[:2], clicks[:2], mean_reward, choose(test_distribution, 1))
