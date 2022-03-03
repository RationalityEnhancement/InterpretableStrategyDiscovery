import argparse
import numpy as np
import pickle
import math
import random
import os
import re
from RL2DT.decision_tree_imitation_learning import train, formula_validation
from RL2DT.strategy_demonstrations import make_modified_env as make_env, \
                                    get_modified_demonstrations as get_demos
from RL2DT.demonstration_sampling import SamplingMachine
from RL2DT.strategy_demonstrations import reward
from RL2DT.elbow_analysis import get_elbows
from RL2DT.formula_visualization import dnf2conj_list as dnf2cnf_list, get_used_preds, check_for_additional_nodes
from RL2DT.mouselab_utils import load_additional_data, join
from RL2DT.hyperparams import BRANCHING, NUM_PROGRAMS, NUM_ACTIONS, ENV_TYPE, ORDER, \
                        BASE_NUM_DEMOS, NUM_CLUSTERS, SPLIT, ASPIRATION_LEVEL, \
                        CLUSTER_DEPTH, MAX_DEPTH, NUM_ROLLOUTS, TOLERANCE
import sys
from .MCRL import modified_mouselab_new
sys.modules['modified_mouselab_new'] =modified_mouselab_new

BLANK = "\n                    " #used for printing purposes
ITERS = 10 #how many times to run the algorithm to establish consistency

formula_values = {}

def random_shuffle(dictionary):
    """
    Shuffle lists that appear in the values of an input dictionary.

    Parameters
    ----------
    dictionary : dict
    """
    for value in dictionary.values():
        if type(value) is list: random.shuffle(value)

def limit(demos_dictionary, to, _from=0):
    """
    Choose (state, action) pairs that came from any demonstrations associated
    with integers in the interval [_from, to).

    Parameters
    ----------
    demos_dictionary : dict
        pos: [ (Any, Any) ]
            Demonstrated (state, action) pairs
        neg: [ (Any, Any) ]
            (state, action) pairs for encountered states and non-optimal actions 
            according to the RL policy
        leng_pos: [ int ]
            Lengths of the consecutive demonstrations
        leng_neg: [ int ]
            Lengths of the negative examples for the consecutive demonstrations
    to : int
    from_ (optional) : int

    Returns
    -------
    output_dictionary : dict
        pos: [ (Any, Any) ]
            Selected (state, action) pairs
        neg: [ (Any, Any) ]
            Selected (state, action) pairs for encountered states and non-optimal 
            actions according to the RL policy
    """
    output_dictionary = {}
    keys = ['pos','neg','leng_pos','leng_neg']
    if not demos_dictionary:
        return None

    num_pos_from = sum(demos_dictionary['leng_pos'][:_from])
    num_neg_from = sum(demos_dictionary['leng_neg'][:_from])
    num_pos_to = sum(demos_dictionary['leng_pos'][_from:to])
    num_neg_to = sum(demos_dictionary['leng_neg'][_from:to])

    output_dictionary['pos'] = demos_dictionary['pos'][num_pos_from:num_pos_to]
    output_dictionary['neg'] = demos_dictionary['neg'][num_neg_from:num_neg_to]
    output_dictionary['leng_neg'] = demos_dictionary['leng_neg'][_from:to] 
    output_dictionary['leng_pos'] = demos_dictionary['leng_pos'][_from:to] 

    return output_dictionary

def label_demos(demos_dictionary, n_clusters):
    """
    Label (state, action) pairs according to the demonstrations they come from.

    Used to enable fast sampling from the demonstration_sampling.SamplingMachine.

    Parameters
    ----------
    demos_dictionary : dict
        pos: [ (Any, Any) ]
            Demonstrated (state, action) pairs
        neg: [ (Any, Any) ]
            (state, action) pairs for encountered states and non-optimal actions 
            according to the RL policy
        leng_pos: [ int ]
            Lengths of the consecutive demonstrations
        leng_neg: [ int ]
            Lengths of the negative examples for the consecutive demonstrations
    n_clusters:
        The number of demonstrations to select from 'demos_dictionary'

    Returns
    -------
    output : [ int ]
        A list of labels for (state, action) pairs in 'demos_dictionary' ordered
        by its keys, and then the original ordering
    """
    output = []
    for i in range(n_clusters):
        num_same_label = demos_dictionary['leng_pos'][i]
        output += [i] * num_same_label

    return output

def load_data(parsed_args, num_demos, double_files=True):
    '''
    Load input data used by the interpretation algorithms

    Files are listed in the Returns section.
    
    Default location is PLP_data folder. Default names are the ones used by
    strategy_demonstrations.py. The only files needed for the algorithms to work
    are demos and expert_reward.

    Parameters
    ----------
    parsed_args : ArgumentParser.Namespace
        argparse parsed input arguments
    num_demos : int
        How many demos to use for interpretation
    double_file : bool
        Whether to add '2' to the names of the files and load more data
        Could be useful if the data is too big to be generated in one sweep
        and there are multiple files with demos of the same kind

    Returns
    -------
    demos : dict
    validation_demos : dict / None
    '''
    args = parsed_args
    cwd = os.getcwd()
    
    if args.custom_data:
        demo_path = args.demo_path
        validation_demo_path = args.validation_demo_path
        
    else:
        env_name = '_' + args.environment if args.environment != 'standard' else ''
        END_PATH = str(BASE_NUM_DEMOS) + env_name + '.pkl'
        demo_path = cwd+'/PLP_data/copycat_' + END_PATH
        validation_demo_path = cwd+'/PLP_data/further_copycat_' + END_PATH
           
    print('Loading the requested files...')

    with open(demo_path, 'rb') as handle:
        demos = pickle.load(handle)

    try:
        with open(validation_demo_path, 'rb') as handle:
            validation_demos = pickle.load(handle)
    except:
        validation_demos = None

    mean_reward = args.mean_reward

    if double_files:
        two = '_2.pkl' if env_name == '' else '2.pkl'

        with open(re.sub('.pkl', two, demo_path), 'rb') as handle:
            add_demos = pickle.load(handle)

        with open(re.sub('.pkl', two, validation_demo_path), 'rb') as handle:
            add_validation_demos = pickle.load(handle)

        demos = join(demos, add_demos)
        validation_demos = join(validation_demos, add_validation_demos)

    if num_demos == -1: num_demos = len(demos['pos'])

    demos = limit(demos, to=num_demos)
    if args.algorithm == 'binary':
        validation_demos = limit(validation_demos, to=num_demos)
    else:
        validation_demos = None
    
    stats = None
    if args.other_stats:
        stats = load_additional_data(parsed_args, num_demos, double_files)

    return demos, validation_demos, mean_reward, stats, demo_path

def wrapper_get_elbows(algorithm, **kwargs):
    '''
    Wrapper function for getting the numbers of clusters
    '''
    if algorithm == 'binary':
        candidate_numbers = ['all']
        dsl_path = os.getcwd() + '/PLP_data/' + kwargs['elbow_dsl_file'] + '.pkl'
    elif kwargs['elbow_method'] == 'automatic':
        candidate_numbers, dsl_path = get_elbows(demo_path=kwargs['demo_path'], 
                                           num_demos=kwargs['num_demos'], 
                                           num_candidates=kwargs['num_candidates'], 
                                           environment=kwargs['env_name'], 
                                           candidate_clusters=kwargs['candidate_clusters'], 
                                           dsl_data=kwargs['elbow_dsl_file'], 
                                           info=kwargs['info'])
    else:
        candidate_numbers = kwargs['candidate_clusters']
        dsl_path = os.getcwd() + '/PLP_data/' + kwargs['elbow_dsl_file'] + '.pkl'
        
    return candidate_numbers, dsl_path

def wrapper_train(tree_depth, demos, validation_demos, data=[None, None]):
    """
    Wrapper function for the 'train' function. 

    See 'decision_tree_imitation_learning.train'.
    """
    return train(program_gen_step_size=500, 
                 num_programs=NUM_PROGRAMS, 
                 num_dts=5, 
                 max_num_particles=25, 
                 input_demos=demos, 
                 further_demos=validation_demos, 
                 tree_depth=tree_depth, 
                 pred_data=data)

def save_binary(obj, aspiration, patience, depth, 
                num_demos, env, num=None, info=''):
    """
    Save the tree computed by interpret_binary in a file.
    """
    cwd = os.getcwd()
    folder_path = cwd+'/interprets/binary_'
    ASPIRATION = '_aspiration_'+str(aspiration)
    PATIENCE = '_patience_'+str(patience)
    DEPTH = '_depth_'+str(depth)
    NDEMO = 'demos_'+str(num_demos)
    ENV_TYPE = '_' + env
    str_n = '_' + str(num) if num is not None else ''
    filename = NDEMO + ENV_TYPE + ASPIRATION + DEPTH + PATIENCE + str_n
    
    add_info = '_' + info if info != '' else ''
    
    with open(folder_path + filename + add_info + '.pkl', 'wb') as handle:
        pickle.dump(obj, handle)
        
def save_adaptive(obj, aspiration, depth, val, val_ratio, 
                  num_demos, num_clusters, env, num=None, info=''):
    """
    Save the tree computed by interpret_adaptive in a file.
    """
    cwd = os.getcwd()
    folder_path = cwd+'/interprets/adaptive_'
    ASPIRATION = '_aspiration_'+str(aspiration)
    DEPTH = '_depth_'+str(depth)
    VALID = '_validation_' + val + '_' + str(val_ratio)
    NDEMO = 'demos_'+str(num_demos)
    NCLUST = '_num_clusters_' + str(num_clusters)
    ENV_TYPE = '_' + env
    str_n = '_' + str(num) if num is not None else ''
    filename = NDEMO + ENV_TYPE + ASPIRATION + DEPTH + VALID + NCLUST + str_n
    
    add_info = '_' + info if info != '' else ''
    
    with open(folder_path + filename + add_info + '.pkl', 'wb') as handle:
        pickle.dump(obj, handle)

## Note we use sampling machine to sample parts of the initial dataset by 
## demonstrations. The whole set is clustered based on demonstrations and 
## num_demos number of the clusters are sampled in total to get the training set. 
## This is equivalent to dividing the demonstrations dataset and running tree 
## learning but saves time as the predicates need to be computed only once, when
## creating the machine.

def interpret_binary(num_demos, aspiration_level, patience, max_depth, 
                     expert_reward, tolerance, num_rollouts, demos=None, 
                     validation_demos=None, dsl_path=None, env_name='', num=None, 
                     info='', **kwargs):
    """
    Try interpreting the behavior of (possibly an RL) policy in the spirit of 
    imitation learning by forming a decision tree through the PLP method from 
    Silver et al. (2019).

    Score the formulas based on the mean reward they obtain on testing environments.

    Cut down the size of the demonstration set if no good formula is found, until 
    a limit.

    Increase the size if a formula is found.

    Changing the size of the demonstration set implements the intuition that 
    with more demonstrations the strategy is richly accounted for, whereas with 
    very few demos, only the most often encountered behavior is captured.

    The more the size changes, the more/the less there are various behaviors of 
    a strategy captured.

    Parameters
    ----------
    num_demos : int
    aspiration_level : float
    patience : int
        The minimum difference of change for the size of the demonstrations set
    max_depth : int
    expert_reward : float
    tolerance : float
    num_rollouts : int
    demos (optional) : dict
    validation_demos (optional) : dict
    dsl_path (optional) = str
    env_name (optional) : str
    num (optional) : int
    info (optional) : str
    
    See interpret_adaptive for explicaiton of the parameters.

    Returns
    -------
    best_formula : StateActionProgram
    """

    prev_num_demos = 2*num_demos
    init_num_demos = num_demos
    increase = 0
    prev_best_formula = None
    found_answer = False

    if demos != None:
        demonstrations = demos 
    else:
        demonstrations = get_demos(demo_numbers=range(num_demos), 
                                   env_branching=BRANCHING)
    if validation_demos != None:
        all_validation_demos = validation_demos
    else:
        all_validation_demos = get_demos(demo_numbers = range(num_demos), 
                                         env_branching = BRANCHING)
        
    print('Setting up the SamplingMachine...')
        
    data_loader = SamplingMachine(demonstrations['pos'], 
                                  demonstrations['neg'], 
                                  use_DSL_features=True,
                                  path_to_DSL_data=dsl_path,
                                  all_data=demos)
    data_loader.cluster(method = label_demos, 
                        demos_dictionary = demos, 
                        n_clusters = init_num_demos)


    while abs(prev_num_demos - num_demos) > patience:

        print(BLANK+BLANK + "Checking demos set of size {}\n".format(num_demos))

        data_loader.reset_pipeline()
        pos_demos, neg_demos, val_demos = sample_sampling_machine(data_loader, 
                                                                  external=False,
                                                                  train_ratio=SPLIT,
                                                                  n_clusters=num_demos,
                                                                  order='random')

        demos_dict = {'pos': pos_demos, 'neg': neg_demos}

        prev_formula_aspiration = -np.inf
        prev_formula_nodes = np.inf
        best_formula = None
        best_aspiration = 0
        best_nodes = np.inf
        
        for dep in range(1, max_depth+1):
            print(BLANK + "Checking formulas with max depth {}\n".format(dep))
            
            formula = wrapper_train(dep, demos_dict, val_demos, 
                                    data=[data_loader.get_X(), 
                                          data_loader.get_y()])
            formula_aspiration = -np.inf
            formula_nodes = get_formula_nodes(formula)

            if best_aspiration > 1-tolerance:
                break

            if formula is not None and (best_aspiration < 1-tolerance or (best_aspiration >= 1-tolerance and formula_nodes < best_nodes)):
                value_expert, value_formula = formula_value_speedup(formula, 
                                                                    expert_reward, 
                                                                    aspiration_level, 
                                                                    tolerance,
                                                                    num_rollouts, 
                                                                    kwargs['stats'],
                                                                    num)
                formula_aspiration =  min(value_expert/value_formula, 
                                          value_formula/value_expert)

            if is_best_formula(formula_aspiration, formula_nodes,
                               prev_formula_aspiration, prev_formula_nodes,
                               aspiration_level, tolerance): 

                best_formula = formula
                best_aspiration = formula_aspiration
                best_demos = num_demos
                best_points = len(pos_demos)
                best_depth = get_formula_depth(formula.program)
                best_nodes = get_formula_nodes(formula)
                best_value = value_formula
                best_stats = (best_points, best_demos, best_depth, 
                              best_nodes, best_value)

                prev_formula_aspiration = formula_aspiration
                prev_formula_nodes = formula_nodes

        print('BEST: {}, PREV: {}'.format(best_formula, prev_best_formula))
        print('Found answer" {}'.format(found_answer))

        if best_formula is None and prev_best_formula is not None:
            found_answer = True
            best_formula = prev_best_formula
            best_stats = prev_best_stats
            break

        print('BEST: {}, PREV: {}'.format(best_formula, prev_best_formula))
        print('Found answer" {}'.format(found_answer))


        if best_formula is not None:
            minimax = int(math.floor(float(prev_num_demos-num_demos) / 2 ))
            increase = minimax
            found_answer = True

            if prev_best_formula and not is_best_formula(best_aspiration, best_nodes,
                                                         prev_best_stats[-1]/mean_reward,
                                                         prev_best_stats[-2], 
                                                         aspiration_level,
                                                         tolerance):
                best_formula = prev_best_formula
                best_stats = prev_best_stats

            if num_demos != init_num_demos:
                num_demos += increase
            else:
                break

            prev_best_formula = best_formula
            prev_best_stats = best_stats

        print('BEST: {}, PREV: {}'.format(best_formula, prev_best_formula))
        print('Found answer" {}'.format(found_answer))

 
        if best_formula is None:
            prev_num_demos = num_demos
            num_demos = int(math.floor(float(num_demos) / 2))

        print('BEST: {}, PREV: {}'.format(best_formula, prev_best_formula))
        print('Found answer" {}'.format(found_answer))


    if found_answer is True:
        all_points = len(demos['pos'])
        print("Best interpretable formula found for {} demos".format(best_stats[1]) + \
              " with {}/{} demos. It has {} nodes, depth {}, return {}".format(best_stats[0],
                                                                               all_points,
                                                                               best_stats[3],
                                                                               best_stats[2],
                                                                               best_stats[-1]))
        print(best_formula)
            
        to_save = {'tree': best_formula.get_tree(), 
                   'program': best_formula.program}
        save_binary(obj=to_save,
                    aspiration=aspiration_level, 
                    patience=patience, 
                    depth=max_depth, 
                    num_demos=num_demos,
                    env=env_name,
                    num=num,
                    info=info)

        return best_formula, best_stats

    else:
        print("The set of predicates is incufficient...")
        return None, None
    
def init_sampling_machine(clustering, num_demos, demos, use_DSL_data, dsl_path,
                          num_clusters, info, **kwargs):
    """
    Initialize sampling object that is either a neural net or a hierarchical 
    tree clustering method, both of which operate on feature vectors from DSL.
    
    Parameters
    ----------
    clustering : str
    num_demos : int
    demos : dict
    use_DSL_data : bool
    dsl_path : str
    num_clusters : int
    info : str
    
    See 'interpret_adaptive' for explication of the paprameters.
        
    Returns
    -------
    sampling_machine : demonstration_sampling.SamplingMachine
        Object performing the clustering of feature vectors
    """
    if clustering == 'hidden_space':
        cwd = os.getcwd()
        folder_path = cwd+'/autoencoder_runs/'
        f = str(kwargs['epochs']) + '_batch_size' + str(kwargs['batch_size']) + \
            '_demo_size' + str(num_demos)
        try:
            with open(folder_path + f, 'rb') as handle:
                checkpoint = pickle.load(handle)
            sampling_machine.load_state_dict(checkpoint['model_state_dict'])
        except:
            sampling_machine = SamplingMachine(demonstrations=demos['pos'], 
                                               negative_demonstrations=demos['neg'], 
                                               lr=kwargs['lr'], 
                                               hidden_size=kwargs['hidden_size'], 
                                               hidden_rep_size=kwargs['hidden_rep_size'], 
                                               num_actions=NUM_ACTIONS,
                                               use_DSL_features=use_DSL_data,
                                               path_to_DSL_data=dsl_path,
                                               all_data=demos)
            sampling_machine.train(kwargs['epochs'], kwargs['batch_size'], num_demos)
        sampling_machine.cluster()

    elif clustering == 'DSL':
        print('Setting up the SamplingMachine...')
        sampling_machine = SamplingMachine(demonstrations=demos['pos'], 
                                           negative_demonstrations=demos['neg'], 
                                           use_DSL_features=True,
                                           path_to_DSL_data=dsl_path,
                                           all_data=demos)
        sampling_machine.cluster(method="hierarchical", 
                                 num_clusters=num_clusters,
                                 vis=True,
                                 info=info)
    return sampling_machine

def sample_sampling_machine(sampling_machine, external, train_ratio, n_clusters, order=ORDER):
    """
    Sample positive an negative demonstrations from the clusters depending on
    whether an external demonstration set is used or we sample from the train set.
    
    Parameters
    ----------
    sampling_machine : demonstration_sampling.SamplingMachine
        Object performing the clustering of feature vectors
    external : bool
        Whether the sample should come from an external set or the train set
    train_ratio : float
        What part of the train set should be used to sample the valiation set
    n_clusters : int
        How many out of all the clusters found y sampling_machine initially are 
        being considered
        
    Returns
    -------
    pos_demos : [ (Any, Any) ]
        Demonstrated (state, action) pairs
    neg_demos : [ (Any, Any) ]
        (state, action) pairs for encountered states and non-optimal actions 
        according to the RL policy
    val_demos : dict
        Dictionary with the same kind of demos as above albeit used only for
        validation
    """
    if not(external):
        res = sampling_machine.sample_from_clusters(num_samples=train_ratio, 
                                                    num_clusters=n_clusters,
                                                    order=order,
                                                    pos_validation=True,
                                                    neg_validation=True,
                                                    clust_max_dep=CLUSTER_DEPTH)
        pos_demos, pos_val_demos = res[0], res[1]
        neg_demos, neg_val_demos  = res[2], res[3]
        val_demos = {'pos': pos_val_demos, 'neg': neg_val_demos}

    else:
        res = sampling_machine.sample_from_clusters(num_samples=1.0, 
                                                    num_clusters=n_clusters,
                                                    order=order)
        pos_demos, neg_demos = res[0], res[1]
        val_demos = validation_demos
        
    return pos_demos, neg_demos, val_demos

def formula_value_speedup(formula, expert_reward, aspiration_level, tolerance,
                          num_rollouts, stats, num):
    """
    Compute estimate return of the policy induced by the formula or extract it
    from the dictionary of already computed values, if it is significantly lower
    than wanted.
    
    Parameters
    ----------
    formula : StateActionProgram
    expert_reward : float
        Value to compare to
    aspiration_level : float
    tolerance : float
    num_rollouts : int
    stats : dict
        Mouselab-specific dictionary with clicks, distributions of clicks etc.
    num : int
        Which iteration is it
        
    Returns
    -------
    value_expert : float
    value_formula : float
    """
    diff_run = False
    
    if formula.program in formula_values.keys():
        value_formula, run = formula_values[formula.program]
        value_expert = expert_reward
        formula_aspiration =  min(value_expert/value_formula, 
                                  value_formula/value_expert)
        if formula_aspiration + tolerance >= aspiration_level and run != num:
            diff_run = True
            
    if formula.program not in formula_values.keys() or diff_run:
        value_expert, value_formula = formula_validation(formula,
                                                         expert_reward,
                                                         num_rollouts,
                                                         stats=stats)
    else:
        print("Value computed earlier in this run or in a different run but yielding " + \
              "insignificant return, repeating: {}".format(value_formula))
    formula_values[formula.program] = (value_formula, num)
    
    return value_expert, value_formula

def is_best_formula(formula_aspiration, formula_nodes, prev_formula_aspiration, 
                    prev_formula_nodes, aspiration_level, tolerance):
    '''
    Establish if the formula achieves significantly higher reward or a similar
    reward but with smaller complexity to decide whether it is better than some
    previous one.
    '''
    
    high_asp = formula_aspiration > prev_formula_aspiration + tolerance
    sim_asp = abs(formula_aspiration - prev_formula_aspiration) <= tolerance
    suff_asp = formula_aspiration + tolerance >= aspiration_level
    less_cmpx = formula_nodes < prev_formula_nodes
            
    if suff_asp and (high_asp or (sim_asp and less_cmpx)):                   
        return True
    return False


def interpret_adaptive(num_demos, aspiration_level, max_depth, tolerance,
                       num_clusters, demos, expert_reward, num_rollouts, validation_demos=None, num_samples=10, clustering='DSL', use_DSL_data=True, dsl_path=None, env_name='', num=None,
                       info='', **kwargs):
    """
    Try interpreting the behavior of (possibly an RL) policy in the spirit of 
    imitation learning by creating a formula through the PLP method from 
    Silver et al. (2019).

    Score the formulas based on the policies they induce and the mean reward 
    those policies obtain on testing environments.

    Sample demonstrations from the clusters, lowering the number of clusters 
    until 1 is hit.
    
    For each numer of clusters, consider all possible depths of the decision tree
    from which the formula is extracted and choose the formula which receives
    the significantly highest reward

    Sampling from clusters affects the representability of the strategy. The more
    clusters are used, the larger scope of the strategy's behavior is captured.

    Parameters
    ----------
    num_demos : int
        Initial number of the demonstrations
    aspiration_level : float
        The minimum proportion of expected reward that should be obtained by the
        policy induced by the formula with respect to the expert's policy 
    max_depth : int
        The maximum depth of the formula to consider
    tolerance : float
        Increase in aspiration value that is considered significant and one 
        formula is preferred over another
    num_clusters : int
        The numer of groups into which the algorithm clusters the binary vectors
    demos : dict
        pos : [ (Any, Any) ]
            Demonstrated (state, action) pairs
        neg : [ (Any, Any) ]
            (state, action) pairs for encountered states and non-optimal actions 
            according to the RL policy
        leng_pos: [ int ]
            Lengths of the consecutive demonstrations
        leng_neg: [ int ]
            Lengths of the negative examples for the consecutive demonstrations
    expert_reward : float
        The list of rewards averaged over 10 000 trials in each generated 
        environment
    num_rollouts : int
        Number of times to initialize a new state and run the policy induced by 
        the output until termination
    validation_demos (optional) : dict
    num_samples (optional) : int or float in [0,1]
        The number or the proportion of samples to draw from each cluster
    clustering (optional) : str
        Preferably a string indicating whether to cluster the data in 'DSL' 
        vector form, compute embedding af the DSL 'hidden_DSL', or compute/load
        a model of a 'hidden_space' 
    use_DSL_data (optional) : bool
        Information specifying if the SamplingMachine uses the binary predicate
        vectors or something else, i.e. a representation found with a neural net
    dsl_path (optional) : str
        A path to a file with the demonstrations written as binary vectors 
        encoding the predicates
    env_name (optional) : str
        Name of the environment for the saving function
    num (optional) : int
        Number identifying the run
    info (optional) : str
        Additional info to add to the name of the file with the found formula

    Returns
    -------
    best_formula : StateActionProgram
    best_stats : tuple
        Used demonstrations, used clusters, depth of the formula, its return
    """

    no_formula = True 
    valid = 'cluster' if not(validation_demos) else 'file'

    assert clustering in ['DSL', 'hidden_space'], \
	"clustering can only be done on 'DSL' data (binary vectors of predicate \
     values) or making use of the 'hidden_space'"
           
    sampling_machine = init_sampling_machine(clustering, num_demos, demos, 
                                             use_DSL_data, dsl_path, num_clusters, 
                                             info, **kwargs)
    n_clusters = sampling_machine.get_nclust()

    start = True
    while n_clusters > 0:

        print(BLANK*2 + "Sampling demos from {} clusters\n".format(n_clusters))
        
        sampling_machine.reset_pipeline()
        pos_demos, neg_demos, val_demos = sample_sampling_machine(sampling_machine, 
                                                                  validation_demos,
                                                                  num_samples,
                                                                  n_clusters)
        sampled_demos = {'pos': pos_demos, 'neg': neg_demos}
        
        if start:
            n_clusters = sampling_machine.get_nclust()
        start = False
        
        prev_formula_aspiration = -np.inf
        prev_formula_nodes = np.inf
        
        for dep in range(1, max_depth+1):
            print(BLANK + "Checking formulas with max depth {}\n".format(dep))
            
            formula = wrapper_train(dep, sampled_demos, val_demos, 
                                    data=[sampling_machine.get_X(), 
                                          sampling_machine.get_y()])
            formula_aspiration = -np.inf
            formula_nodes = get_formula_nodes(formula)

            if formula is not None:
                value_expert, value_formula = formula_value_speedup(formula, 
                                                                    expert_reward, aspiration_level, tolerance,
                                                                    num_rollouts, 
                                                                    kwargs['stats'],
                                                                    num)
                formula_aspiration =  min(value_expert/value_formula, 
                                          value_formula/value_expert)
            
            if is_best_formula(formula_aspiration, formula_nodes,
                               prev_formula_aspiration, prev_formula_nodes,
                               aspiration_level, tolerance):                   
                no_formula = False
                best_formula = formula
                best_points = len(pos_demos) + len(val_demos['pos'])
                best_depth = get_formula_depth(formula.program)
                best_nodes = get_formula_nodes(formula)
                best_clusters = n_clusters
                best_value = value_formula
                best_stats = (best_points, best_clusters, best_depth, 
                              best_nodes, best_value)
                
                prev_formula_aspiration = formula_aspiration
                prev_formula_nodes = formula_nodes
         
        if no_formula is False:
            break
        else:
            n_clusters -= 1

    if no_formula is False:
        all_points = len(demos['pos'])
        print("Best interpretable formula found for {} clusters".format(best_clusters) + \
              " with {}/{} demos. It has {} nodes, depth {}, return {}".format(best_points,
                                                                               all_points,
                                                                               best_nodes,
                                                                               best_depth,
                                                                               best_value))
        print(best_formula)
        
        val_ratio = round(1 - num_samples, 2) if num_samples<1 else num_samples
        to_save = {'tree': best_formula.get_tree(), 
                   'program': best_formula.program}
        save_adaptive(obj=to_save, 
                      aspiration=aspiration_level, 
                      depth=max_depth, 
                      val=valid, 
                      val_ratio=val_ratio, 
                      num_demos=num_demos,
                      num_clusters=num_clusters, 
                      env=env_name, 
                      num=num, 
                      info=info)
        return best_formula, best_stats
     
    else:
        print("The set of predicates is insufficient...")
        return None, None

def get_formula_depth(formula):
    '''
    Compute the depth of a formula (maximum number of predicates in a conjunction).
    
    Parameters
    ----------
    formula : str
        String representing the formula

    Returns
    -------
    formula_depth : int
        Depth of the formula
    '''
    stripped_formula = formula[2:-2]
    subformulas = stripped_formula.split('or')
    depths = []
    for subf in subformulas:
        nodes = subf.split(') and ')
        depths.append(len(nodes))
    formula_depth = max(depths)
    return formula_depth

def get_formula_nodes(formula):
    '''
    Compute the number of distinct predicates used in the formula.
    
    Parameters
    ----------
    formula : StateActionProgram

    Returns
    -------
    n_nodes : int
        Number of distinct predicates/number of nodes in a decision tree equivalent
        to the formula
    '''
    if not formula:
        return np.inf
    program = formula.program
    preds_in_cnfs = dnf2cnf_list(program)
    set_preds = get_used_preds(preds_in_cnfs)
    preds_in_cnfs, set_preds = check_for_additional_nodes(preds_in_cnfs, 
                                                          set_preds)
    n_nodes = len(set_preds)
    return n_nodes

def choose_interpretable(data_by_clusters):
    '''
    Select the most interpretable tree among those outputted for different numbers
    of clusters.
    
    The most interpretable is the one with the fewest nodes and then with the 
    lowest depth.

    Parameters
    ----------
    data_by_clusters : dict
        int : dict
            Number of clusters : statistics associated with the output

    Returns
    -------
    formula : str
        String representing the formula
    stats : dict
        Statistics associated with the formula
    '''
    complexity = np.inf
    depth = np.inf
    best = None
    for k, v in data_by_clusters.items():
        if None in v.values():
            continue
        elif (v['nodes'] < complexity) or \
             (v['nodes'] == complexity and v['depth'] < depth):
            complexity = v['nodes']
            depth = v['depth']
            best = k
            
    if best:
        formula = data_by_clusters[best]['formula']
        stats = {}
        for key in ['points', 'clusters', 'depth', 'nodes', 'return']:
            stats[key] = data_by_clusters[best][key]
        stats['all_clusters'] = best
    else:
        formula = None
        stats = None
    
    return formula, stats

def create_or_expand(dict_, key, element):
    '''
    Ad-hoc function which adds the most interpretable formulas along with their 
    stats, found in each run of the algorithm, to a dictionary.
    '''
    if key in dict_.keys():
        dict_[key]['times'] += 1
        if element is not None:
            for interkey in ['points', 'clusters', 'all_clusters', 'return']:
                dict_[key][interkey] += [element[interkey]]
    else:
        dict_[key] = {'times': 1}
        if element is not None:
            dict_[key] = {'times': 1, 'points': [element['points']], 
                          'clusters': [element['clusters']], 
                          'all_clusters': [element['all_clusters']],
                          'depth': element['depth'], 'nodes': element['nodes'], 
                          'return': [element['return']]}
    return dict_

def wrapper_interpret(algorithm, **kwargs):
    '''
    Wrapper function for running the interpretation algorithm
    '''
    interpret={'binary': interpret_binary, 'adaptive': interpret_adaptive}
    formula_statistics = {}
    candidate_numbers, dsl_path = wrapper_get_elbows(algorithm, **kwargs)
      
    for i in range(ITERS):
        print("\n\n" + BLANK*2 + "ITERATION {}\n\n".format(i))
        data_by_clusters = {}
        for n_cl in candidate_numbers:
            print("\n\n\n\n" + BLANK*2 + "       " + \
                  "CHECKING {} AS THE NUMBER OF CLUSTERS\n\n\n\n".format(n_cl))
            output, output_st = interpret[algorithm](**kwargs, num_clusters=n_cl, 
                                                     dsl_path=dsl_path, num=i)
            formula = output.program if output else None
            stats = output_st if output_st else [None]*5
            data_by_clusters[n_cl] = {'formula': formula, 'points': stats[0], 
                                      'clusters': stats[1], 'depth': stats[2], 
                                      'nodes': stats[3], 'return': stats[4]}
        interpretable_formula, statistics = choose_interpretable(data_by_clusters)
        formula_statistics = create_or_expand(formula_statistics,
                                              interpretable_formula,
                                              statistics)
    return formula_statistics


if __name__ == "__main__":
    
    cwd = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', '-a', 
                        choices={'binary', 'adaptive'},
                        help="Name of the interpretation algorithm to use.", 
                        default='adaptive')
    parser.add_argument('--clustering_type', '-ct',
                        choices={'DSL', 'hidden_space'},
                        help="Whether to cluster DSL data hierarhically or " \
                            +"use a neural network and learn a hidden space.",
                        default='DSL')
    parser.add_argument('--elbow_choice', '-elb',
                        choices={'automatic','manual'},
                        help="Whether to find the candidates for the number of " \
                            +"clusters automatically or use the " \
                            +"candidate_clusters parameter.",
                        default='automatic')
    parser.add_argument('--candidate_clusters', '-cl',
                        type=int, nargs='+',
                        help="The candidate(s) for the number of clusters in " \
                            +"the data either to consider in their entirety " \
                            +"or to automatically choose from.",
                        default=NUM_CLUSTERS)
    parser.add_argument('--num_candidates', '-nc',
                        type=int,
                        help="The number of candidates for the number of " \
                            +"clusters to consider.",
                        default=4)
    parser.add_argument('--custom_data', '-cd',
                        type=bool,
                        help="Whether to use the input data or load from the " \
                            +"PLP_data folder.", 
                        default=False)
    parser.add_argument('--demo_path', '-dp',
                        type=str,
                        help="Path to the file with the demonstrations.", 
                        default='')
    parser.add_argument('--validation_demo_path', '-v',
                        type=str,
                        help="Path to the file with the validation demonstrations.", 
                        default='')
    parser.add_argument('--other_stats', '-os',
                        type=bool,
                        help="Whether to compute state-action visitation stats.", 
                        default=False)
    parser.add_argument('--mean_reward', '-mr',
                        type=float,
                        help="Mean reward of the expert.")
    parser.add_argument('--num_demos', '-n', 
                        type=int,
                        help="How many demos to use for interpretation.", 
                        default=-1)
    parser.add_argument('--name_dsl_data', '-dsl', 
                        type=str,
                        help="Name of the .pkl file containing input demos turned " \
                            +"to binary vectors of predicates in folder PLP_data", 
                        default=None)
    parser.add_argument('--interpret_size', '-i',
                        type=int,
                        help="Maximum depth of the interpretation tree",
                        default=MAX_DEPTH)
    parser.add_argument('--aspiration_level', '-l',
                        type=float,
                        help="How close should the intepration performance in " \
                            +"terms of the reward be to the policy's performance",
                        default=ASPIRATION_LEVEL)
    parser.add_argument('--tolerance', '-t',
                        type=float,
                        help="What increase in the percentage of the expected " \
                             "expert reward a formula is achieving is " \
                             "considered significant when comparing a two of them.",
                        default=TOLERANCE)
    parser.add_argument('--num_rollouts', '-rl',
                        type=int,
                        help="How many rolouts to perform to compute mean " \
                            +"return per environment",
                        default=NUM_ROLLOUTS)
    parser.add_argument('--samples', '-s',
                        type=float,
                        help="How many samples/in what ratio to sample from clusters",
                        default=SPLIT)
    parser.add_argument('--patience', '-p',
                        type=int,
                        help="See 'patience' parameter for the binary algorithm",
                        default=4)
    parser.add_argument('--environment', '-e',
                        type=str,
                        help="Name of the environment for which files in the " \
                            +"PLP_data folder exist",
                        default=ENV_TYPE)
    parser.add_argument('--info', '-f',
                        type=str,
                        help="What to add to the name of all the output files",
                        default='')

    args = parser.parse_args()

    res = load_data(args, args.num_demos, double_files=not(args.custom_data))
    demos, validation_demos, mean_reward, stats, d_path = res
   
    data_by_clusters = wrapper_interpret(algorithm=args.algorithm, 
                                         num_demos=args.num_demos,
                                         env_name=args.environment,
                                         aspiration_level=args.aspiration_level,
                                         patience=args.patience,
                                         max_depth=args.interpret_size,
                                         tolerance=args.tolerance,
                                         demo_path=d_path,
                                         demos=demos,
                                         validation_demos=validation_demos,
                                         expert_reward=mean_reward,
                                         num_rollouts=args.num_rollouts,
                                         elbow_dsl_file=args.name_dsl_data,
                                         num_samples=args.samples,
                                         elbow_method=args.elbow_choice,
                                         clustering=args.clustering_type,
                                         num_candidates=args.num_candidates,
                                         candidate_clusters=args.candidate_clusters,
                                         info=args.info,
                                         stats=stats)
    
    print('\n\n FORMULAS:\n')
    for formula, data in data_by_clusters.items():
        print('\n\n  {}:'.format(formula))
        for k,v in data.items():
            print('{}: {}\n'.format(k,v))
