from formula_procedure_transformation import ConversionError, DNF2LTL, trajectorize
from hyperparams import ALLOWED_PREDS, REDUNDANT_TYPES
from RL2DT.hyperparams import *
from translation import alternative_procedures2text as translate
import argparse
import pickle
import os
from toolz import curry
import copy
from RL2DT.formula_visualization import dnf2conj_list as d2l, prettify
from geneticalgorithm import geneticalgorithm as ga
from formula_procedure_transformation import size, sizeBig
from evaluate_procedure import compute_log_likelihood_demos
from functools import lru_cache
import numpy as np
import sys
    

def represent_modular(formula, raw_formula):
    """
    Represent the input procedural formula in a list of steps.
    
    Parameters
    ----------
    formula : str
        Procedural formula 
    formula_raw : str
        Procdeural formula written in a callable form
        
    Returns
    -------
    modular_representation : [ dict ]
        dict
            conjunction : str
                Conjunction of predicates encoding a step of the procedural formula
            until : str
                Until condition for the step 
            unless : str
                Unless condition for the step
            conjunction_raw : str
                Same as above but wirtten in callable form 
            until_raw : str
                Same as above but wirtten in callable form 
            unless_raw : str
                Same as above but wirtten in callable form 
            go_to : str
                Conjunction to return to after completing all the steps
            go_to_raw : str
                Same as above but wirtten in callable form
    ids : [ str ]
        List of unique predicates uesed in the procedural description
    """
    modular_representation = []
    ids = []
    def id_predicates(conj):
        preds = d2l(conj, paran=False)[0]
        for p in preds:
            if p not in ids:
                ids.append(p)
    
    parts = formula.split(' AND NEXT ')
    parts_raw = raw_formula.split(' AND NEXT ')
    go_to, go_to_raw = '', ''
    if 'LOOP' in parts[-1]:
        last_el, go_to = parts[-1].split('\n\n')
        last_el_raw, go_to_raw = parts_raw[-1].split('\n\n')
        parts[-1] = last_el
        parts_raw[-1] = last_el_raw
        
    for index, (p, pr) in enumerate(zip(parts, parts_raw)):
        unless_p, unless_pr = '', ''
        first_p, first_pr = p, pr
        until_p, until_pr = '', ''
        if 'UNLESS' in p:
            first_p, unless_p = p.split(' UNLESS ')
            first_pr, unless_pr = pr.split(' UNLESS ')
        if 'UNTIL' in p:
            first_p, until_p = first_p.split(' UNTIL ')
            first_pr, until_pr = first_pr.split(' UNTIL ')
        if first_pr[:2] == '((':
            first_pr = first_pr[2:]
        if first_pr[-2:] == '))' and first_pr[-5:] != 'act))' and \
           first_pr[-3:] != ' ))':
            first_pr = first_pr[:-2]
        elif first_pr[-3:] == ')) ' and first_pr[-5:] != 'act)) ' and \
             first_pr[-4:] != ' )) ':
            first_pr = first_pr[:-3]
        id_predicates(first_p)
        
        dict_ = {'conjunction': first_p, 'until': until_p, 'unless': unless_p,
                 'conjunction_raw': first_pr, 'until_raw': until_pr, 
                 'unless_raw': unless_pr}
        modular_representation.append(dict_.copy())
        
    modular_representation.append({'go_to': go_to, 'go_to_raw': go_to_raw})
    return modular_representation, ids

def count_predicates(formula, raw_formula):
    """
    Determine the number of unique predicates appearing in the input procedural
    formula.
    
    Parameters
    ----------
    formula : str
        Procedural formula 
    formula_raw : str
        Procdeural formula written in a callable form
        
    Returns
    -------
    num : int
        Number of unique predicates in 'formula'
    """
    _, ids = represent_modular(formula, raw_formula)
    num = len(ids)
    return num

def select_subformula(modular_representation, predicate_ids, binary_vector):
    """
    Output a procedural subformula of a formula represented in modular_representation
    by selecting only the predicates which were not removed according to the
    binary_vector.
    
    Parameters
    ----------
    (See represent_modular)
    modular_representation : [ dict ]
    predicate_ids : [ str ]
    binary_vector : np.array
        vec.shape = (1, len(modular_representation)-1)
        Binary vector representing the steps included in a procedural formula
        divided into separate elements of modular_representation
    
    Returns
    -------
    formula : str
        Procedural subformula 
    formula_raw : str
        Procdeural subformula written in a callable form
    """
    formula, formula_raw = '', ''
    loop_exists = False
    go_to = modular_representation[-1]
    g = go_to['go_to']
    g_raw = go_to['go_to_raw']
    if g!= '': 
        g = g[10:] #remove LOOP FROM
        g_raw = g_raw[10:] #remove LOOP FROM
    g_until, g_until_raw = '', ''
    prev = ''
    if 'UNLESS' in g:
        g, g_until = g.split(' UNLESS ')
        g_raw, g_until_raw = g_raw.split(' UNLESS ')
        g_until = ' UNLESS ' + g_until
        g_until_raw = ' UNLESS ' + g_until_raw
    last_until = False
        
    for ind, mod in enumerate(modular_representation[:-1]):
        conj = d2l(mod['conjunction'], paran=False)[0]
        conj_raw = d2l(mod['conjunction_raw'], paran=False)[0]
        current, current_raw = '', ''
        loop_formula = False

        if mod['conjunction'] == g:
            loop_formula = True
            loop_exists = True
            goto, goto_raw = '', ''
        all_zero = True
        for c, cr in zip(conj, conj_raw):
            vector_index = predicate_ids.index(c)
            if binary_vector[vector_index] == 1:
                if all_zero:
                    current += c
                    current_raw += cr
                    if loop_formula:
                        goto += c
                        goto_raw += cr
                else:
                    current += ' and ' + c
                    current_raw += ' and ' + cr
                    if loop_formula:
                        goto += ' and ' + c
                        goto_raw += ' and ' + cr
                all_zero = False
                
        if all_zero: #whole conjunction removed = always True
            current += 'True'
            current_raw += 'True'
            if loop_formula:
                goto += 'True'
                goto_raw += 'True'

        if current != prev: #different predicate after AND NEXT than before
            formula += current
            formula_raw += current_raw
            last_until = False
            
            if mod['until'] != '':
                formula += ' UNTIL ' + mod['until']
                formula_raw += ' UNTIL ' + mod['until_raw']
                last_until = True
            
            if mod['unless'] != '':
                formula += ' UNLESS ' + mod['unless']
                formula_raw += ' UNLESS ' + mod['unless_raw']
                last_until = True
            
            formula += ' AND NEXT '
            formula_raw += ' AND NEXT '
            
        else:
            len_until = 2 if ' or ' in mod['until'] else 1 if mod['until'] != '' else 0
            len_unless = 2 if ' or ' in mod['unless'] else 1 if mod['unless'] != '' else 0
        
        if current == 'True' and current!= prev and \
           'STOPS APPLYING' in mod['until']: #instruction denoting clicking 
                                             #randomly and terminating; the rest
                                             #of the procedure isn't important
            loop_exists = False
            break
        prev = current
        
    formula = formula[:-10]
    formula_raw = formula_raw[:-10]
    
    if loop_exists:
        if current == goto:
            if g_until == '': 
                if not(last_until):
                    formula += ' UNTIL IT STOPS APPLYING'
                    formula_raw += ' UNTIL IT STOPS APPLYING'
            else:
                if not(last_until):
                    formula += ' UNTIL ' + g_until[8:]
                    formula_raw += ' UNTIL ' + g_until_raw[8:]
        else:
            formula += '\n\nLOOP FROM ' + goto + g_until
            formula_raw += '\n\nLOOP FROM ' + goto_raw + g_until_raw
            
    return formula, formula_raw


def progress(count, total, status=''):
    """
    Function that displays a loading bar.
    
    Parameters
    ----------
    count : int
    total : int
    status : str (optional)
    """
    bar_len = 50
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '|' * filled_len + '_' * (bar_len - filled_len)

    sys.stdout.write('\r%s %s%s %s' % (bar, percents, '%', status))
    sys.stdout.flush() 

def greedy_check(function, dimension):
    """
    Search for the optimal (minimum) solution of a binary funciton 
                   'function' : {0,1}^dimension -> IR 
    by doing a greedy search.
    
    Start with a vector <1,...,1> and switch 1 to 0 if it improves the value of
    'function'. Repeat 'dimension' times to try all the possibilities.
    
    Parameters
    ----------
    function : callable
        Pythonic function that accepts a binary vector as an input and returns a
        real value
    dimension : int
        Dimension of 'function''s domain
    
    Returns
    -------
    vec : np.array
        vec.shape = (1, dimension)
        Greedy-optimal binary vector minimizing the value of 'function'
    best_val : float
        Greedy-minimal value
    """
    vec = np.ones(dimension)
    best_val = function(vec)
    print('\nInitial value: {}\n'.format(best_val))
    counter = 0
    for _ in range(len(vec)):
        for pos, _ in enumerate(vec):
            counter += 1
            progress(counter, (len(vec))**2, status="Greedy algorithm is running...")
            test_vec = vec.copy()
            test_vec[pos] = 0
            val = function(test_vec)
            print('\nVec: {}, val: {}'.format(test_vec, val))
            if val < best_val:
                best_val = val
                vec = test_vec
    print('\nGreedy vector found: {}\n\nwith value: {}\n'.format(vec, best_val))
    return vec, best_val

@curry
def quality_function(binary_vector, modular_rep, pred_ids, demos, pipeline):
    """
    Evaluate the quality of a binary vector.
    
    Vector represents which steps of a procedural formula represented in modular_rep
    (divided by NEXT steps) are to be considered, and which not.
    
    Quality is the number of removed steps minus the log marginal likelihood of the 
    pruned formula under the demonstrations (better vectors are the smaller ones and 
    those which fit the data well).
    
    Parameters
    ----------
    binary_vector : np.array
        vec.shape = (1, len(modular_rep)-1)
        Binary vector representing the steps included in a procedural formula
        divided into separate elements of modular_rep
    (See represent_modular)
    modular_rep : [ dict ]
    pred_ids : [ str ]
    demos : ( [ [ int ] ], [ [ int ] ] )
        List of environments encoded as rewards hidden under the nodes; List of 
        action sequences for each consecutive environment
    pipeline : [ ( [ int ], function ) ]
        List of parameters used for creating the Mouselab MDP environments: 
        branching of the MDP and the reward function for specifying the numbers
        hidden by the nodes.
    
    Returns
    -------
    score : float
        Computed quality of the input vector
    """
    
    selected_formula, selected_formula_raw = select_subformula(modular_rep, 
                                                               pred_ids,
                                                               binary_vector)
    envs, action_seqs = demos
    
    res = compute_log_likelihood_demos(envs=tuple(tuple(e) for e in envs), 
                                       pipeline=(tuple(pipeline[0][0]), pipeline[0][1]), 
                                       acts=tuple(tuple(a) for a in action_seqs), 
                                       formula=selected_formula_raw,
                                       verbose=False)
    ll, lmarglik, opt_score, ml, opt_act_score, e_1 = res
    
    score = lmarglik - sum(binary_vector)
    return -score #genetic algorithm minimizes by default

def simplify_and_optimize(formula, formula_raw, envs, action_seqs, pipeline, algorithm='greedy'):
    """
    Prune the input formula to finds its simpler representation which achieves 
    higher or equal marginal likelihood.
    
    Parameters
    ----------
    formula : str
        Procedural formula 
    formula_raw : str
        Procdeural formula written in a callable form
    envs : [ [ int ] ]
        List of environments encoded as rewards hidden under the nodes; index of
        the reward corresponds to the id of the node in the Mouselab MDP
    action_seqs : [ [ int ] ]
        List of action sequences for each consecutive environment
    pipeline : [ ( [ int ], function ) ]
        List of parameters used for creating the Mouselab MDP environments: 
        branching of the MDP and the reward function for specifying the numbers
        hidden by the nodes.
    algorithm : str (optional)
        Whether to perform 'ga' genetic algorithm optimization or use the 'greedy'
        method
    
    Returns
    -------
    new_formula : str 
        Best procedural subformula of the input procedural formula according to
        the marginal likelihood
    new_formula_raw : str
        Same formula as above written in a callable form
    lmarglik : float
        Log marginal likelihood of the policy induced by the pruned procedural 
        formula
    """
    assert algorithm in ['greedy', 'ga'], "algorithm needs to be 'greedy' or 'ga'" + \
        " not {}".format(algorithm)

    p_data = (envs, action_seqs)
    modular_formula_rep, predicate_ids = represent_modular(formula, formula_raw)
    
    ga_vec_size = count_predicates(formula, formula_raw)
    ga_q_func = quality_function(modular_rep=modular_formula_rep,
                                 pred_ids=predicate_ids,
                                 demos=p_data,
                                 pipeline=pipeline)
    ga_varbound=None #for Boolean type
    ga_params={'max_num_iteration': None,'population_size': 100, 
               'mutation_probability': 0.5, 'elit_ratio': 0.01, 
               'crossover_probability': 0.5, 'parents_portion': 0.3,
               'crossover_type':'uniform', 'max_iteration_without_improv': None}
    model=ga(function=ga_q_func,
             function_timeout=20,
             dimension=ga_vec_size,
             variable_type='bool',
             variable_boundaries=ga_varbound,
             convergence_curve=False,
             algorithm_parameters=ga_params)
    
    if algorithm == 'greedy':
        best_binary_vec, val = greedy_check(function=ga_q_func,
                                           dimension=ga_vec_size)
    elif algorithm == 'ga':
        print('Genetic algorithm is running...')
        model.run()
        best_binary_vec = model.best_variable
    
    new_formula, new_formula_raw = select_subformula(modular_formula_rep,
                                                     predicate_ids,
                                                     best_binary_vec)
    lmarglik = -val + sum(best_binary_vec)
    
    return new_formula, new_formula_raw, lmarglik
