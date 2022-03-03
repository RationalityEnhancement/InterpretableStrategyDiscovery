from random import shuffle, randint
from collections import defaultdict
from RL2DT.formula_visualization import dnf2conj_list as d2l, prettify
from RL2DT.PLP.DSL import *
from RL2DT.decision_tree_imitation_learning import get_program_set
from RL2DT.interpret import join
from RL2DT.strategy_demonstrations import reward
from RL2DT.hyperparams import TERM
from hyperparams import THRESHOLD
from evaluate_procedure import compute_log_likelihood_demos
import os
import re
import argparse
import pickle
import copy
import itertools
import random

class IncompatibilityError(Exception):
    """
    Exception when the number of failures reaches the maximum.
    """
    def __init__(self, num_failures, max_failures):
        self.num_failures = num_failures
        self.max_failures = max_failures
        self.message = "Too many demonstraions for which the until " +\
                       "predicate is non-existent: " +\
                       "{} for maximum of {}".format(self.num_failures, 
                                                     self.max_failures)
        super().__init__(self.message)
        
        
class ConversionError(Exception):
    """
    Exception when the conversion of a modified formula to a procedural formula 
    fails.
    """
    def __init__(self):
        self.message = "The formula cannot be transformed after removing " +\
                       "the redundant predicates."
        super().__init__(self.message)

def find_always(phi):
    """
    Find a common prefix of all conjunctions in a DNF phi.

    Parameters
    ----------
    phi : str
        DNF formula
        
    Returns
    -------
    common_prefix : str
        Prefix (a conjunction of predicates) that is part of every conjunction
        in phi
    """
    preds_in_alternatives = d2l(phi)
    longest_path = 0
    for i, path in enumerate(preds_in_alternatives):
        if len(path) > longest_path:
            longest_pred = preds_in_alternatives[i]
            longest_path = len(path)

    prefixes = [longest_pred[:i+1] for i in range(longest_path)]
    common_prefix = None
    for ksi in prefixes:
        pref = ksi
        for dnf in preds_in_alternatives:
            is_prefix = [pref[i] == dnf[i] for i in range(len(pref))]
            if not all(is_prefix):
                return common_prefix
        common_prefix = ksi
    return common_prefix

def best_until(pred, until_conds, raw_phi, p_envs, p_actions, p_pipeline, unless=False):
    """
    Choose a predicate for the UNTIL condition.
    
    Choose the first predicate among until_conds which contains some of the
    predicates from pred. Otherwise, choose the simplest predicate.

    Parameters
    ----------
    pred : str
        Formula, conjunction of predicates
    until_conds : [ str ]
        Candidates for the UNTIL predicate
    raw_phi : str
        Bigger formula to have the until conditions be evaluated on
    p_envs : [ [ int ] ]
        List of environments encoded as rewards hidden under the nodes; index of
        the reward corresponds to the id of the node in the Mouselab MDP
    p_actions : [ [ int ] ]
        List of action sequences for each consecutive environment
    p_pipeline : [ ( [ int ], function ) ]
        List of parameters used for creating the Mouselab MDP environments: 
        branching of the MDP and the reward function for specifying the numbers
        hidden by the nodes.
    unless : bool (optional)
        
    Returns
    -------
    until : str
        Chosen UNTIL predicate
    """
    for u in until_conds:
        simp_pred = '((' + simplify2(pred, u) + '))'
        if simp_pred != pred:
            return u
    
    best_lik = -np.inf
    until_conds2 = [u for u in until_conds if ' or ' not in u]
    until_conds = until_conds2 if until_conds2 != [] else until_conds
    for ind, u in enumerate(until_conds):
        if not(unless):
            tst_raw = raw_phi + pred + ' UNTIL ' + u + \
                      ' AND NEXT True UNTIL IT STOPS APPLYING'
        else:
            if 'LOOP FROM' in raw_phi:
                add = ''
            else:
                add = ' AND NEXT True UNTIL IT STOPS APPLYING'
            tst_raw = raw_phi + ' UNLESS ' + u + add
            
        print('TESTING: {}/{}'.format(ind+1,len(until_conds)), end='\r')
                      
        res = compute_log_likelihood_demos(tuple(tuple(e) for e in p_envs), 
                                        pipeline=(tuple(p_pipeline[0][0]), p_pipeline[0][1]), 
                                        acts=tuple(tuple(a) for a in p_actions), 
                                        formula=tst_raw,
                                        verbose=False)
        if res[0] > best_lik:
            best_u = u
            best_lik = res[0]

    return best_u

def biggest_intersection_most_elts(subsets, elements, up_to):
    """
    Given different sets, output a subset of elements appearing in those sets
    that are the most frequent (~ first part of the "Apriori" algorithm)

    Parameters
    ----------
    subsets : [ [str ] ]
        Sets of certain elements (here, predicates)
    elements : [ str ]
        Set of "outputable" elements (here, predicates)
    up_to : int
        Tolerance parameter which specifies how many times an element might NOT
        appear in one of the subsets and still be returned
        
    Returns
    -------
    intersection : [ str ]
        Subset of elements which appears in the most number of sets from subsets
        (here, predicates)
    """
    d = dict()
    for subs in subsets:
        visited = {k: False for k in elements}
        for s in subs:
            keys = [el for el in elements if s in el]
            for key in keys:
                if not(visited[key]):
                    d[key]=d.setdefault(key, 0)+1
                    visited[key] = True
    m = max(d.values()) if d != dict() else 0
    if m < len(subsets) - up_to:
        raise IncompatibilityError(len(subsets)-m, up_to)
    else:
        intersection = [el for el in d.keys() if d[el] == m]
    return intersection

def find_until(phi, trajs, whens, index, predicate_matrix, predicates, 
               allowed_predicates, ignored):
    """
    Determine the UNTIL/UNLESS condition for appyling formula phi.
    
    Search each separate trajectory and find the set of predicates which were 
    constantly 0 for a part of this trajectory when phi was True, and then, 
    when phi stopped being True, turned to 1.
    
    Select these among the found predicates, which are allowed (a hyperparameter
    set by hand).
    
    Return a subset of those predicates which are the most frequent among the 
    trajectories.

    Parameters
    ----------
    phi : str
        DNF formula
    trajs : [ [ ( IHP.modified_mouselab.Trial, int ) ] ]
        Lists of state-action pairs from rollouts
    whens : [ [ int ] ]
        List of lists of indices of the state-action pairs where one of the 
        conjunctions of the DNF formula phi stopped being True
    index : int
        Index which specifies which switch between the conjunctions (evidenced 
        in whens) is being considered
    predicate_matrix : scipy.sparse.csr_matrix
        Matrix of binary vectors where each row corresponds to a state-action
        pair from trajs
    predicates : dict
        int : RL2DT.PLP.policy.StateActionProgram
        Predicate id : PLP version of the predicate itself
    allowed_predicates : [ str ]
        Candidates for the UNITL/UNLESS predicate
    ignored : [ int ]
        Indices of the set of all state-action pairs for which the DNF formula
        phi was False
        
    Returns
    -------
    pred_candidates : [ str ]
        Best candidates for the UNTIL/UNLESS predicate
    """
    candidates = set()
    all_points = [el for el in trajs]
    bad_threshold = THRESHOLD * len(all_points)
    bad = 0
    trajs_allowed_cands = []
    print('Searching for the until/unless predicate...')
    for i, ex in enumerate(trajs):
        ex_candidates = set()
        if index >= len(whens[i]):
            #print("\nUnfortunate example, moving on...\n")
            continue
        w = whens[i][index]

        start = 0 if (index == 0 or len(whens[i]) == -index) \
                  else whens[i][index]-whens[i][index-1]
        subtraj = ex[start:]
        #print([({n.label: n.value for n in subtraj[k][0].observed_nodes}, 
        #        subtraj[k][1]) for k in range(len(subtraj))])
        if index + 1 < len(whens[i]):
            end = whens[i][index+1] - whens[i][index] # correcition for the case 
                                                      # when we have beta1, 
                                                      # unknown strategy, beta2
        else:
            end = 0
            for state, action in subtraj:
                if not eval('lambda st, act : ' + phi)(state, action):
                    break
                end += 1
            end += whens[i][index]
        sum_vec = predicate_matrix[w].astype(int)

        for num, ex_vec in enumerate(predicate_matrix[w+1:w+end]):
            if w+1+num not in ignored:
                sum_vec = map(sum, zip(sum_vec, ex_vec.astype(int)))
        for j, pred_val in enumerate(sum_vec):
            #if pred_val == end and predicate_matrix[w+end][j] == 0:
            #    ex_candidates.add(j)
            if pred_val == 0 and predicate_matrix[w+end][j] == 1:
                ex_candidates.add(j)
        pred_ex_cands = [predicates[can].program for can in ex_candidates]
        allowed_ex_cands = set(allowed_predicates).intersection(set(pred_ex_cands))
        trajs_allowed_cands.append(allowed_ex_cands)
        
    pred_candidates = biggest_intersection_most_elts(trajs_allowed_cands,
                                                     allowed_predicates, 
                                                     bad_threshold)
    return pred_candidates

def which_holds(alternatives, state, action):
    """
    Return the first formula from alternatives that is satisfied with state and
    action as its arguments

    Parameters
    ----------
    alternatives : [ str ]
        List of formulas
    state : any
    action : int
        
    Returns
    -------
    alt : str
        First satisfied formula
    """
    for alt in alternatives:
        func_alt = eval('lambda st, act : ' + alt)
        if func_alt(state, action):
            return alt
        
def remove_prefix(phi, prefix):
    """
    Remove prefix from conjunction phi.

    Parameters
    ----------
    phi : str
        Conjunction of predicates
    prefix : [ str ]
        Prefix to remove
        
    Returns
    -------
    suffix : str
        phi without the input prefix
    """
    preds = d2l(phi)[0]
    leave = len(prefix) if prefix else 0
    suffix = ' and '.join(preds[leave:])
    return suffix

def extract_disjoint_formula(reference, psi):
    """
    Assuming reference and psi are two conjunctions of predicates extracted from
    a decision tree, return that part of psi, which is disjoint from reference
    in the tree (e.g. if the tree is S and ref= SAB, psi= Snot(A)C, then beta= C
                                  yes|
                                     A
                                yes / \ no
                                   B   C

    Parameters
    ----------
    reference : str
        Conjunction of predicates to compare psi to
    psi : str
        Conjunction of predicates
        
    Returns
    -------
    beta : str
        Part of psi that does not follow the same branch of the decision tree it
        was extracted from
    """
    preds_ref = d2l(reference)[0]
    preds_psi = d2l(psi)[0]
    for num, (p_ref, p_psi) in enumerate(zip(preds_ref, preds_psi)):
        if not(p_psi == 'not(' + p_ref + ')' or p_ref == 'not(' + p_psi + ')'):
            start_num = num
            break
    beta = ' and '.join(preds_psi[start_num:])
    return beta

def no_redundancy(preds, redundant_preds, keywords=None):
    """
    Check if neither of the redundant_preds or their negation is in preds and
    that neither of the keywords appear in predicates in preds.

    Parameters
    ----------
    preds : [ str ]
        List of predicates
    redundant_preds : [ str ]
        List of unwanted predicates
    keywords (optional) : None or [ str ]
        List of unwanted keywords
        
    Returns
    -------
    bool
        Whether neither of the preds nor the keywords can be found in preds
    """
    for p in preds:
        for rp in redundant_preds:
            if p == 'not(' + rp + ')' or rp == 'not(' + p + ')' or p == rp:
                return False
            elif keywords != None and any([k in p for k in keywords]):
                return False
    return True

def remove_preds(preds, psi, keywords=None):
    """
    Remove predicates, negation of predicates from preds or predicates that
    contain any of they keywords from conjunction psi.

    Parameters
    ----------
    preds : [ str ]
       Predicates which are to be removed from psi
    psi : str
        Conjunction of predicates
    keywords (optional) : None or [ str ]
        List of unwanted keywords
        
    Returns
    -------
    psi : str
        Input conjunction without the unwanted predicates
    """
    if keywords == None: keywords = []
    preds_psi = d2l(psi)[0]
    to_keep = [p for p in preds_psi if not(any([k in p for k in keywords]) or \
               p in preds or p[4:-1] in preds)]
    psi = ' and '.join(to_keep)
    return psi

def unique(sequence):
    """
    Strip a sequence of elements out of repetitions while keeping the ordering.

    Parameters
    ----------
    sequence : [ any ]
        Sequence of elements
        
    Returns
    -------
    ret : [ any ] 
        List of unique elements in the same order as appearing in sequence
    """
    seen = set()
    ret = [x for x in sequence if not (x in seen or seen.add(x))]
    return ret

def temporalize(phi, trajs, predicate_matrix, redundant_preds, pred_types=None, 
                no_redunancy=True):
    """
    Use the trajectories to find the temporal characteristic of the input DNF
    formula.
    
    The idea is that the trajectories are iterated and we check which of the
    conjunctions in the input DNF formula is satisfied, then stick to it, and 
    move to another conjunction after the previous one stops being True. The 
    moments when this happens are used to determine the UNTIL/UNLESS conditions.

    Parameters
    ----------
    phi : str
        DNF formula
    trajs : [ [ ( IHP.modified_mouselab.Trial, int ) ] ]
        Lists of state-action pairs from rollouts
    predicate_matrix : scipy.sparse.csr_matrix
        Matrix of binary vectors where each row corresponds to a state-action
        pair from trajs
    redundant_preds : [ str ]
        List of unwanted predicates
    pred_types (optional) : None or [ str ]
        Keyword appearing in unwanted predicates
    no_redunancy : bool
        Whether phi consists of some of the redundant_preds or predicates which
        contain some of the pred_types
        
    Returns
    -------
    M : [ [ str ] ]
        For each trajectory, sequence of conjunctions from phi that are True
        across the state-action pairs in the trajectory. This sequence satisfies
        that while iterating through the trajectory, firstly the first conjunction
        is satisfied, then the 2nd, and so on, until the last is satisfied and
        the trajectory ends
    L : [ [int ] ]
        For each trajectory, number of state-action pairs in that trajectory when
        a singular conjunction from the DNF phi evaluated to True (before another
        conjunction evaluated to True or the trajectory finished)
    W : [ [ int ] ]
        For each trajectory, global indices (counting for the first state-action
        pair in the first trajectory) of the state-action pairs when one of the 
        conjunctions from the DNF phi stopped being True
    C : dict
        str : [ str ]
        Dictionary with keys being the conjunctions in phi and values being these
        among the conjunctions from phi, which are True after the key becomes 
        False somewhere in at least on of the input trajectories. Only some of
        the conjunctions which satisfy this requirement appear in the value
    I : [ int ]
        Indices of the set of all state-action pairs for which the DNF formula
        phi is False
    alpha : str
        Common prefix of all conjunctions in phi
    """
    alpha = find_always(phi)
    alternatives = phi.split(' or ')
    n = len(trajs)
    phi_new = False
    M, L, W, I = [[]] * n, [[]] * n, [[]] * n, []
    C = {}
    counter = -1
    print('Dividing demos based on the used strategy...')
    
    for i, ex in enumerate(trajs):
        #print('\n')
        #print([(['{}: {}'.format(n.label, n.value) 
        #         for n in t.observed_nodes], a) 
        #         for t,a in ex])
        L_ex, W_ex = [], []
        seq = []
        phi_new = False
        for num, (state, action) in enumerate(ex):
            counter += 1
            if action == TERM: ##comes last
                L_ex += [l]
                if not(no_redunancy):
                    W_ex += [counter]
                    if beta not in C.keys(): C[beta] = []
                    continue
            if eval('lambda st, act : ' + phi)(state, action) == False:
                I.append(counter)
                continue
            if phi_new == False:
                phi_new = which_holds(alternatives, state, action)
                beta = remove_preds(preds=redundant_preds, psi=phi_new, 
                                    keywords=pred_types)
                seq = [beta]
                l = 1
                W_ex += [counter]
            elif eval('lambda st, act : ' + phi_new)(state, action) == False:
                phi_prev = phi_new
                phi_new = which_holds(alternatives, state, action)
                prev_beta = beta
                disjoint = remove_preds(preds=redundant_preds, psi=phi_new, 
                                        keywords=pred_types)
                beta = disjoint
                if prev_beta not in C.keys() and beta != prev_beta:
                    C[prev_beta] = [beta]
                elif beta != prev_beta:
                    C[prev_beta].append(beta)
                W_ex += [counter]
                L_ex += [l]
                l = 1
                seq += [beta]
            else:
                l += 1
        M[i] = seq
        L[i] = L_ex
        W[i] = W_ex
    
    C = {key: list(unique(val)) for key, val in C.items()}
    return M, L, W, C, I, alpha

## bc2 = {'a': ['b', 'c', 'd'], 'b': ['c', 'd'], 'c': ['e'], 'e': ['b']}

def get_paths(beta, beta_connections, visited):
    """
    Find all possible sequences of elements from beta_connections that start with
    beta and end with :
      -> an element that does not have connections or
      -> with an element which signifies a connection to an element which has 
         already appeared before.

    Parameters
    ----------
    beta : str
        Sequence of elements
    betas_connections : dict
        str : [ str ]
        Dictionary which specifies connections between the elements
    visited : [ str ]
        Elements that have been already considered
        
    Returns
    -------
    res : [ [ ... [ str ] ... ] ] 
        List of arbitrarily nested values from betas_conections that correspond
        to sequences of these element that are connected to one another and end
        with an element which does not have connections or with an element that
        connects back to one of the already visited elements. See "bc2" above.
    """
    if 'LOOP' in beta and beta[10:] in visited:
        return [beta]
    beta = beta[10:] if "LOOP" in beta else beta
    if beta not in beta_connections.keys() or beta_connections[beta] == []:
        return [beta]
    res = []
    visited.append(beta)
    for b in beta_connections[beta]:
        beta_visited = copy.deepcopy(visited)
        res += [[beta] + get_paths(str(b), beta_connections, beta_visited)]
    return res

def any_connects(to_what, start, beta_connections, visited=[]):
    """
    Modify beta_connections so that each to_what appearing in the values of this
    dictionary was changed to LOOP FROM to_what.
    
    Do that only when this particular to_what can be accessed via connections 
    specified by beta_connections beginning with element start.

    Parameters
    ----------
    to_what : str
        Element a sequence is to end with
    start : str
        Element to start inspecting betas_connections from
    betas_connections : dict
        str : [ str ]
        Dictionary which specifies connections between the elements
    visited (optional) : [ str ]
        Elements that have been already considered
        
    Returns
    -------
    betas_connections : dict
        str : [ str ]
        Modified input dictionary where the following mapping was performed: 
        to_what |-> LOOP FROM to_what
        whenever to_what can be accesed through some connections starting with 
        element start
    """
    start = start[10:] if "LOOP" in start else start
    if start not in beta_connections.keys():
        return "not in connections"
    betas = beta_connections[start]
    if start in visited:
        return "visited"
    elif to_what in betas:
        visited.append(start)
        beta_connections[start] = [b if b != to_what else "LOOP FROM " + str(to_what) 
                                   for b in beta_connections[start]]
    visited.append(start)
    bool_or_list = [any_connects(to_what,b, beta_connections,visited) for b in betas]
    
    if all([b == 'visited' for b in bool_or_list]):
        return beta_connections

def rollout_paths(beta_connections):
    """
    Modify beta_connections so that it was possible to generate a sequence
    el_1, el_2, ..., el_n, LOOP FROM el_i for some i < n using the elements
    of the modified beta_connections and any starting point.

    Parameters
    ----------
    betas_connections : dict
        str : [ str ]
        Dictionary which specifies connections between the elements
        
    Returns
    -------
    betas_connections : dict
        str : [ str ]
        Modified input dictionary where the following mapping was performed: 
        el |-> LOOP FROM el
        whenever el can be access itself via connection specified in 
        beta_connections
    """
    for beta in beta_connections.keys():
        any_connects(beta, beta, beta_connections, [])
    return beta_connections
        
def flatten_paths(paths, type_=str):
    """
    Flatten each nested list in the input paths in a way defined by flatten_path. 
    
    Only for lists with elements of the same type.

    Parameters
    ----------
    paths : [ [ ... [ any ] ... ] ]
        List of arbitrarily nested lists
    type_ (optional) : any
        Type of the primitives in the lists
        
    Returns
    -------
    flat_paths : [ [ any ] ]
        List of non-nested versions of the lists from the input
    """
    if len(paths) == 1 and type(paths[0]) is type_:
        return [paths]
    flat_paths = []
    for path in paths:
            if len(path) == 2 and type(path[0]) is type_ and type(path[1]) is type_:
                flat_paths += [path]
            else:
                flat_paths += flatten_path([], path)[0]
    return flat_paths
    
def flatten_path(fpaths, path, type_=str):
    """
    Flatten a nested list treating the level of nesting as a point when the flat
    lists should branch out. For example [a, [[b,c], d]] |-> [[a,b,c], [a,b,d]].
    
    Only for a list with elements of the same type.

    Parameters
    ----------
    fpaths : [ [ any ] ]
        Current flat lists
    path : [ [ ... [ any ] ... ] ]
        Nested list to be flattened
    type_ (optional) : any
        Type of the primitives in the lists
        
    Returns
    -------
    newpaths : [ [ any ] ]
        Flattened lists residing in the input list path
    bool
        Whether the list was initially flat or not
    """
    if len(path) == 2 and type(path[0]) is type_ and type(path[1]) is type_:
        return path, True
    else:
        rets = []
        for p in path[1:]:
            end, b = flatten_path(fpaths, p)
            if b: rets.append(end)
            else: rets += end
        newpaths = []
        for r in rets:
            newpaths.append([path[0]] + r)
        return newpaths, False

def synthesize(paths):
    """
    Output a list of lists from paths for which neither is fully included in any
    other.
    
    Extend the lists, if some of the connections in others allow that.

    Parameters
    ----------
    paths : [ [ any ] ]
        List of lists
        
    Returns
    -------
    synthesized_paths : [ [ any ] ]
        Sublist of paths where neither of the elements is a proper subset of 
        another
    """
    synthesized_paths = []
    for i in range(len(paths)):
        bad = False
        for j in range(len(paths)):
            if i != j:
                if is_included(paths[i], paths[j]):
                    bad = True
        if not bad: synthesized_paths.append(paths[i])
        
    new_synth_paths = []
    no_loops = [s[:-1] if "LOOP" in s[-1] else s for s in synthesized_paths]
    uniques = list(set(tuple(row) for row in no_loops))
    uniques = [list(el) for el in uniques]
    clusters = []
    for u in uniques:
        indices = ()
        for ind, nl in enumerate(no_loops):
            if nl == u:
                indices += (ind,)
        clusters.append(indices)
    loops = [s[-1] if "LOOP" in s[-1] else "" for s in synthesized_paths]
    loop_clusters = []
    for c in clusters:
        loops_c = tuple(loops[i] for i in c)
        loop_clusters.append(loops_c)
    for ind, u in enumerate(uniques):
        cands = loop_clusters[ind]
        if len(cands) == 1:
            new_synth_paths.append(u + [cands[0]])
        else:
            earliest_ind = np.inf
            for c in cands:
                index = u.index(c[10:])
                if index < earliest_ind:
                    add_loop = c
            new_synth_paths.append(u + [add_loop])
            
    synthesized_paths = new_synth_paths
                    
    return synthesized_paths
            
def is_included(s1, s2):
    """
    Check if s1 is included in s2 so that higher the index of an element in s1, 
    the higher it is in s2. In other words, the ordering is kept.

    Parameters
    ----------
    s1 : [ any ]
        List of elements
    s2 : [ any ]
        List of elements
        
    Returns
    -------
    bool
        Whether the s1 is a subset of s2 with the same ordering as in s2
    """
    search_ind = 0
    for e in s1:
        broke = False
        for i in range(search_ind, len(s2)):
            if e == s2[i]:
                search_ind = i
                broke = True
                break
        if not broke: return False
    return True

def is_included_loop(el, loop_format_rep):
    """
    Rollout a sequence of elements of the format a_1, ..., (a_i, ..., a_n)*.
    
    Check if el is included in the rollouted sequence. See is_included. 

    Parameters
    ----------
    el : [ any ]
        List of elements
    loop_format_rep : [ any, str ]
        List of elements ending with LOOP FROM el for some element el in the list
        
    Returns
    -------
    bool
        Whether the s1 is a subset of s2 with the same ordering as in s2
    """
    copy_rep = copy.deepcopy(loop_format_rep)
    if "LOOP" in loop_format_rep[-1]:
        repeat = loop_format_rep[-1][10:]
        ind_repeat = loop_format_rep.index(repeat)
        copy_rep = copy.deepcopy(loop_format_rep)[:-1]
        #while len(el) >= len(copy_rep[:-1]):
        #    copy_rep += copy_rep[ind_repeat:]
        for _ in range(len(el)):
            copy_rep += copy_rep[ind_repeat:]
    return is_included(el, copy_rep)

def compact(betas_sequences, betas_lengths, betas_whens):
    """
    Compact a list of elements and its accompanying list with statistics (numbers
    associated with each element) to join together neighboring duplicates.
    
    Add the lenghts and make a list out of whens.

    Parameters
    ----------
    betas_sequences : [ [ str ] ]
        List of sequences
    betas_lengths : [ [ int ] ]
        List of numbers corresponding to each element in each of the sequences
    betas_whens : [ [ int ] ]
        List of numbers corresponding to each element in each of the sequences
        
    Returns
    -------
    (See above)
    new_betas_sequences : [ [ str ] ]
    new_betas_lengths : [ [ int ] ]
    new_betas_whens : [ [ int ] ]
    """
    new_betas_sequences = []
    new_betas_lengths = []
    new_betas_whens = []
    
    def list_neighboring_duplicates(seq):
        i = 0
        tally = defaultdict(list)
        full_pass = False
        while not(full_pass):
            for j in range(i, len(seq)):
                if (j in tally.keys() or j in tally.values()):
                    full_pass = True
                    break
                if seq[j] == seq[i]:
                    tally[i].append(j)
                else:
                    i = j
                    break
        return tally
            
    for i, bs in enumerate(betas_sequences):
        tally = list_neighboring_duplicates(bs)
        new_betas_seq = [bs[j] for j in tally.keys()]
        new_betas_len = [sum([betas_lengths[i][j] for j in dups]) 
                         for dups in tally.values()]
        new_betas_when = [betas_whens[i][j] for j in tally.keys()]
        if len(betas_whens[i]) > len(bs):
            new_betas_when.append(betas_whens[i][-1])
        new_betas_sequences.append(new_betas_seq)
        new_betas_lengths.append(new_betas_len)
        new_betas_whens.append(new_betas_when)
    return new_betas_sequences, new_betas_lengths, new_betas_whens

class EquivalenceClass:
    """
    For representing equivalence relation as being a subsequence of some given
    sequence.
    
    Sequence needs to have a form a_1, ..., (a_i, ..., a_n)* and be repreesnted
    as a list [a_1, ..., a_i, ..., a_n, LOOP FROM a_i]
    
    Contains additional statistics associated with the elements in the same 
    equivalence class: lens and whens (see L and W in temporalize).
    """
    def __init__(self, elts = [], rep = [], lens = [], whens = [], traj_elts=[]):
        self.elements = elts
        self.whens = whens
        self.traj_elements = traj_elts
        self.first = rep[0]
        if "LOOP" in rep[-1]:
            self.loop = [rep[-1]]
            self.representation = rep[:-1]
        else:
            self.loop = ['']
            self.representation = rep
        if lens != []:
            assert len(lens) == len(self.representation), \
                   "Len has to be equal to the size of rep"
        self.lengths = lens if lens != [] else [0 for r in self.representation]
        self.element_counter = [0 for r in self.representation]
        self.full_representation = self.representation + self.loop
        
    def _update_length(self, el, len_el):
        for i, val in enumerate(el):
            for j, rep_val in enumerate(self.representation):
                if val == rep_val:
                    self.lengths[j] += len_el[i]
                    self.element_counter[j] += 1
                            
    def append(self, new_el, lens_el, when_el, traj_el):
        self.elements.append(new_el)
        self.whens.append(when_el)
        self.traj_elements.append(traj_el)
        self._update_length(new_el, lens_el)
        
    def update(self):
        self.lengths = [float(val) / num if num != 0 else 0 
                        for val, num in zip(self.lengths, self.element_counter)]
    
    def set_loop(self, el):
        self.loop = [el]
        self.full_representation = self.representation + self.loop
        
    def transitioned_whens(self, ind):
        ret=[w for w in self.whens if len(w[ind:]) > 1]
        return ret
    
    def transitioned_traj_elements(self, ind):
        return [self.traj_elements[i] for i in range(len(self.whens)) 
                if len(self.whens[i][ind:]) > 1]
        
    def ending_with_beta_whens_traj_elements(self, beta):
        chosen_whens = []
        chosen_traj_elts = []
        for i, el in enumerate(self.elements):
            if beta[2:-2] == el[-1]:
                chosen_whens.append(self.whens[i])
                chosen_traj_elts.append(self.traj_elements[i])
        return chosen_whens, chosen_traj_elts

def unroll_connections(connections, starts, max_len):
    """
    Find sequences defining equivalence classes (according to the EquivalenceClass)
    by considering which sequences can be created using beta_connections.
    
    Find simple sequences defining equivalence classes of the form 
    a_1, ..., (a_i, ..., a_n)*.
    
    Create more complex sequences by considering two sequences that loop from a_i
    and add combine prefix of one with the suffix of another, e.g. 
    a_1, ..., (a_i, ..., a_n)* + b_1, ..., (a_i, b_i+1, ..., b_m)*  =
    a_1, ..., a_i, ..., a_n, a_i, b_i+1, ..., b_m
    as long as the length isn't above max_len.

    Parameters
    ----------
    betas_connections : dict
        str : [ str ]
        Dictionary which specifies connections between the elements
    starts : [ str ]
        Initial elements to construct sequences starting in
    max_len : int
        Maximum length for a sequence defining an equivalence class
        
    Returns
    -------
    new_equiv_classes : [ EquivalenceClass ]
        List of equivalence classes found by rollouting sequences according to 
        betas_connections (See EquivalenceClass); without the elements of these
        classes
    """
    candidates = []
    connections = rollout_paths(connections)
    for key in starts:
        paths = synthesize(
                    flatten_paths(
                        get_paths(key, connections, [])))
        candidates += paths
    representations = synthesize(candidates)
    equiv_classes = [EquivalenceClass(rep = repp, elts=[], whens=[], 
                                      lens=[], traj_elts=[]) for repp in representations]
    firsts = list(unique([e.first for e in equiv_classes]))
    new_equiv_classes = copy.deepcopy(equiv_classes)
    for f in firsts:
        loops = list(unique([e.loop[0] for e in equiv_classes if e.first == f]))
        for l in loops:
            if l == '':
                continue
            building_blocks_left = [e.representation for e in equiv_classes 
                                    if e.first == f and e.loop == [l]]
            building_blocks_right_l = [e.representation for e in equiv_classes 
                                       if e.first == f and e.loop != ['']]
            building_blocks_right_nol = [e.representation for e in equiv_classes 
                                         if e.first == f and e.loop == ['']]
            combined_l = list(itertools.product(building_blocks_left,
                                                building_blocks_right_l))
            combined_nol = list(itertools.product(building_blocks_left,
                                                  building_blocks_right_nol))
            combined_l = [a+b for a,b in combined_l if a != b]
            combined_nol = [a+b for a,b in combined_nol if a != b]
            combined_loop = [c + [l] for c in combined_l]
            current_reps = [e.full_representation for e in new_equiv_classes]
            combined_all = []
            for el in combined_nol+combined_loop:
                if el not in combined_all:
                    combined_all.append(el) 
            new_equiv_classes += [EquivalenceClass(rep = repp, elts=[], whens=[], 
                                                   lens=[], traj_elts=[]) 
                                  for repp in combined_all]
            while not all([len(rep) >= max_len for rep in combined_nol]):
                combined_l = list(itertools.product(combined_l, 
                                                    building_blocks_right_l))
                combined_nol = list(itertools.product(combined_l,
                                                      building_blocks_right_nol))
                combined_l = [a+b for a,b in combined_l if a != b]
                combined_nol = [a[0]+b for a,b in combined_nol if a[0] != b]
                combined_loop = [c + [l] for c in combined_l]
                combined_all = []
                for el in combined_nol+combined_loop:
                    if el not in combined_all:
                        combined_all.append(el) 
                new_equiv_classes += [EquivalenceClass(rep = repp, elts=[], 
                                                       whens=[], lens=[], 
                                                       traj_elts=[]) 
                                      for repp in combined_all]
    new_reps = [e.full_representation for e in new_equiv_classes]
    synth_reps = synthesize(new_reps)
    new_equiv_classes = [EquivalenceClass(rep = repp, elts=[], whens=[],
                                          lens=[], traj_elts=[]) for repp in synth_reps]
    return new_equiv_classes

def trajs_quotient(betas_sequences, betas_lengths, betas_connections, betas_whens,
                   betas_starts, trajs, verbose=True):
    """
    Find equvalence classes of sequences defined by betas_connections and assign
    sequences from betas_sequences to them.

    Parameters
    ----------
    (See unroll_connections)
    betas_connections : dict
        str : [ str ]
    betas_starts : [ str ]
    (See compact)
    betas_sequences : [ [ str ] ]
    betas_lengths : [ [ int ] ]
    betas_whens : [ [ int ] ]
    trajs : [ [ ( any, any ) ] ]
        Lists of state-action pairs from rollouts of some policy
    verbose (optional) : bool
        Whether to print out sequence representations of the found equivalence
        classes
        
    Returns
    -------
    equiv_classes : [ EquivalenceClass ]
        List of equivalence classes found by rollouting sequences according to 
        betas_connections (See EquivalenceClass); each with elements from
        betas_sequences
    """
    print('Finding executions of the procedure encoded in the DNF formula' + \
          ' based on the connections between its different branches revealed by' + \
          ' the demonstrations.')
    n = len(betas_sequences)
    limit = len(betas_sequences)
    max_len = max([len(seq) for seq in betas_sequences])
    equiv_classes = unroll_connections(betas_connections, betas_starts, max_len)
    if verbose:
        for e in equiv_classes:
            for et in e.representation+e.loop:
                print(et)
            print("\n")
    for i in range(n):
        belongingness = [is_included_loop(betas_sequences[i], e.representation+e.loop) 
                         for e in equiv_classes]
        try:
            #cls = belongingness.index(True) ## the culprit; adds an elem to 1 rep
            for cls, belong in enumerate(belongingness):
                if belong:
                    equiv_classes[cls].append(betas_sequences[i], betas_lengths[i], 
                                              betas_whens[i], trajs[i])
        except:
            if verbose: print('Procedural representation-incompatible demo...')
            limit -= 1
    equiv_classes = [e for e in equiv_classes if len(e.elements)>0]
    [e.update() for e in equiv_classes]
    if limit < int(n/2) or equiv_classes == []:
        return None
    return equiv_classes

def size(sub_formula):
    """
    Output number of predicates appearing in sub_formula. among and all_ get a 
    special treatment.

    Parameters
    ----------
    sub_formula : str
        String representing a logical formula
        
    Returns
    -------
    formula_size : int
        Number of predicates in sub_formula
    """
    if sub_formula == '' or 'IT' in sub_formula:
        return 0
    sub_formula = prettify(sub_formula)
    if ' or ' in sub_formula:
        return 2
    double = sub_formula[:2] == '(('
    while not(double):
        sub_formula = '(' + sub_formula + ')'
        double = sub_formula[:2] == '(('
    preds = d2l(sub_formula)[0]
    formula_size = 0
    for p in preds:
        if ' and ' in p:
            formula_size += 2
        else:
            formula_size += 1
        if ':' in p:
            formula_size += 1
        if 'among' in p or 'all_' in p:
            formula_size += 1
    return formula_size

def sizeBig(formula):
    """
    Output number of predicates appearing in a procedural formula.

    Parameters
    ----------
    formula : str
        String representing a procedural formula
        
    Returns
    -------
    size_f : int
        Number of predicates in the formula. False and True are trated as 1.
    """
    body_loop = formula.split('\n\nLOOP FROM ')
    body = body_loop[0]
    loop = ''
    if len(body_loop) == 2:
        loop = body_loop[1]
    parts = body.split(' AND NEXT ')
    size_f = 0
    for p in parts:
        b_u = p.split(' UNLESS ')
        u = '' if len(b_u) < 2 else b_u[1]
        b = b_u[0]
        b_u = b.split(' UNTIL ')
        u2 = '' if len(b_u) < 2 else b_u[1]
        b = b_u[0]
        size_f += size(b) + size(u2) + size(u)
    size_f += size(loop)
    return size_f

def DNF2LTL(phi, trajs, predicate_matrix, allowed_predicates, p_envs, p_actions,
            p_pipeline, redundant_predicates=[], redundant_predicate_types=None):
    """
    Create a procedural version of an input DNF formula using the demonstrations
    as sources of temporal dynamics.
    
    The procedural formula needs to be of the form
      pred_1 UNTIL derp_1 (UNLESS repd_1) (AND NEXT pred_2) (UNTIL derp_2) ...
      (LOOP FROM pred_i)
    
    If no derp_i can be found, the UNTIL condition becomes UNTIL IT STOPS APPLYING 
    and regards pred_i.
    
    LOOP FROM pred_i means "go to the previous occurrence of pred_i in the 
    formula".
    
    High-level idea is that each conjunction of the DNF is treated as a module
    that could become pred_i.
    
    By iterating the trajectories and checking which module is True in each step
    of the trajectory, we find connections between the modules -- how they switch
    their truth values so that the whole DNF is still True.
    
    For example for a DNF with 2 conjunctions we can find that every trajectory
    starts with conj_1 True and then, after some number of steps, conj_1 becomes
    False and conj_2 start being True. Then the found procedural formula would
    be conj_1 UNTIL something AND NEXT conj_2.
    
    UNTIL and UNLESS predicates ('something' in the example above) are found by
    checking which among the allowed_predicates changed from being constantly
    False to being True following the same dynamics as for the change in the
    modules. 
    
    For example, 'something' would be found by iterating through the trajectories 
    and checking which of the predicates was False when conj_1 was True, and then 
    changed to True, when conj_1 changed to False.

    Parameters
    ----------
    phi : str
        DNF formula
    trajs : [ [ ( IHP.modified_mouselab.Trial, int ) ] ]
        Lists of state-action pairs from rollouts
    predicate_matrix : scipy.sparse.csr_matrix
        Matrix of binary vectors where each row corresponds to a state-action
        pair from trajs
    allowed_predicates : [ str ]
        Candidates for the UNITL/UNLESS predicate (selected a priori by hand to
        create a set of sensible predicates)
    p_envs : [ [ int ] ]
        List of environments encoded as rewards hidden under the nodes; index of
        the reward corresponds to the id of the node in the Mouselab MDP
    p_actions : [ [ int ] ]
        List of action sequences for each consecutive environment
    p_pipeline : [ ( [ int ], function ) ]
        List of parameters used for creating the Mouselab MDP environments: 
        branching of the MDP and the reward function for specifying the numbers
        hidden by the nodes.
    redundant_predicates (optional) : [ str ]
        List of unwanted predicates
    redundant_predicate_types (optional) : None or [ str ]
        Keyword appearing in unwanted predicates
        
    Returns
    -------
    phi_all : str
        Procedural formula/program in readable form
    complexity : int
        Complexity (the number of used predicates) in the description
    raw_phi_all : str
        Procedural formula/program in callable form (with lambdas, colons, etc.)
    """
    if phi == None:
        print("The formula was not found, i.e. allows all kinds of planning.")
        return "None (treated as fully random planning)", 1, "True UNTIL IT STOPS APPLYING"
    preds_in_dnfs = d2l(phi)
    all_preds = [p for dnf in preds_in_dnfs for p in dnf]
    noRedundancy = no_redundancy(all_preds, 
                                 redundant_predicates, 
                                 redundant_predicate_types)
    if (len(preds_in_dnfs) == 1 and noRedundancy) or \
       (len(preds_in_dnfs) == 1 and len(preds_in_dnfs[0])) == 1:
        print("The formula remains the same across all timesteps, " + \
              "trivially returning identity.")
        u = " UNTIL IT STOPS APPLYING"
        if '((' in phi:
            return prettify(phi[2:-2])+u, size(phi[2:-2]), phi[2:-2]+u
        return prettify(phi)+u, size(phi), phi+u
    
    M, L, W, C, I, alpha = temporalize(phi, trajs, predicate_matrix, 
                                       redundant_predicates, 
                                       redundant_predicate_types,
                                       noRedundancy)
    starts = list(unique([seq[0] for seq in M]))
    M, L, W = compact(betas_sequences=M, betas_lengths=L, betas_whens=W)
    equiv_classes = trajs_quotient(betas_sequences=M, 
                                   betas_lengths=L, 
                                   betas_connections=C,
                                   betas_whens=W,
                                   betas_starts=starts,
                                   trajs=trajs)
    if equiv_classes is None:
        print("The formula cannot be represented as a procedural program. " + \
              "Returning the DNF.")
        parts = phi.split(' or ')
        output, output_raw = '', ''
        num_preds = 0
        for p in parts:
            output += prettify(p) + " UNTIL IT STOPS APPLYING" + '\n\nOR\n\n'
            num_preds += size(p)
            output_raw += p + " UNTIL IT STOPS APPLYING" + '\n\nOR\n\n'
        return output[:-6], num_preds, output_raw[:-6]

    phi_all = ''
    raw_phi_all = ''
    complexity = 0
    preds = get_program_set(predicate_matrix.shape[1])[0]
    prev = -100
    
    for num, eqc in enumerate(equiv_classes):
        if num >= 1000:
            break
        print('\n{}: Find a procedural description'.format(num))
        phi_fin = ''
        raw_phi_fin = ''
        for i, beta in enumerate(eqc.full_representation):
            if beta == '': 
                break
            if beta[:2] != '((':
                beta = '((' + beta + '))'
                
            if i == 0:
                if eqc.lengths[i] > 1:
                    try:
                        untils = find_until(phi=beta, 
                                            trajs=eqc.transitioned_traj_elements(i),
                                            whens=eqc.transitioned_whens(i), 
                                            index=i, 
                                            predicate_matrix=predicate_matrix, 
                                            predicates=preds, 
                                            allowed_predicates=allowed_predicates, 
                                            ignored=I)
                        until = best_until(beta, untils, raw_phi_fin, 
                                           p_envs, p_actions, p_pipeline)
                        new_beta = simplify2(beta, until)
                        complexity += size(new_beta)
                        new_beta_p = prettify(new_beta)
                        if eqc.loop != [] and eqc.loop[0] == "LOOP FROM " + beta:
                            eqc.set_loop ("LOOP FROM " + new_beta)
                        phi_fin = new_beta_p + ' UNTIL ' + prettify(until)
                        raw_phi_fin = new_beta + ' UNTIL ' + until
                        complexity += size(until)
                    except IncompatibilityError:
                        if noRedundancy: 
                            phi_fin += prettify(beta) + ' UNTIL IT STOPS APPLYING '
                            raw_phi_fin += beta + ' UNTIL IT STOPS APPLYING '
                            complexity += size(beta)
                        else:
                            raise ConversionError
                else:
                    phi_fin = prettify(beta)
                    raw_phi_fin = beta
                    complexity += size(beta)
                    
            elif i+1 <= len(eqc.representation):
                if eqc.lengths[i] > 1:
                    try:
                        untils = find_until(phi=beta, 
                                            trajs=eqc.transitioned_traj_elements(i),
                                            whens=eqc.transitioned_whens(i), 
                                            index=i, 
                                            predicate_matrix=predicate_matrix, 
                                            predicates=preds, 
                                            allowed_predicates=allowed_predicates, 
                                            ignored=I)
                        until = best_until(beta, untils, raw_phi_fin+' AND NEXT ',
                                           p_envs, p_actions, p_pipeline)
                        new_beta = simplify2(beta, until)
                        complexity += size(new_beta)
                        new_beta_p = prettify(new_beta)
                        if eqc.loop != [] and eqc.loop[0] == "LOOP FROM " + beta:
                            eqc.set_loop ("LOOP FROM " + new_beta)
                        phi_fin += ' AND NEXT ' + new_beta_p + ' UNTIL ' + \
                                   prettify(until)
                        raw_phi_fin += ' AND NEXT ' + new_beta + ' UNTIL ' + until
                        complexity += size(until)
                    except:
                        if noRedundancy: 
                            phi_fin += ' AND NEXT ' + prettify(beta) + \
                                       ' UNTIL IT STOPS APPLYING'
                            raw_phi_fin += ' AND NEXT ' + beta + ' UNTIL IT STOPS APPLYING'
                            complexity += size(beta)
                        else:
                            raise ConversionError
                else:
                    phi_fin += ' AND NEXT ' + prettify(beta)
                    raw_phi_fin += ' AND NEXT ' + beta
                    complexity += size(beta)
                    
            else:
                if eqc.loop[0] != '':
                    phi_fin += ' \n\n' + prettify(eqc.loop[0])
                    raw_phi_fin += ' \n\n' + eqc.loop[0]
                    complexity += size(eqc.loop[0][10:])
                    
            if len(eqc.representation) > 1:
                if prev == i-1:
                    unless = best_until(beta, unlesses, raw_phi_fin, p_envs, 
                                        p_actions, p_pipeline, unless=True)
                    phi_fin += ' UNLESS ' + prettify(unless)
                    raw_phi_fin += ' UNLESS ' + unless
                    complexity += size(unless)
                if "LOOP" not in beta:
                    whens, traj_elts = eqc.ending_with_beta_whens_traj_elements(beta)
                    if whens != [] and len(traj_elts) != len(eqc.traj_elements):
                        try: 
                            ind = -1 if noRedundancy else -2
                            unlesses = find_until(beta, traj_elts, whens, ind, 
                                                  predicate_matrix, preds,
                                                  allowed_predicates, I)
                            prev = i
                        except:
                            unlesses = ['True']

        phi_all += '\n\nOR\n\n' + phi_fin if num != 0 else phi_fin
        raw_phi_all += '\n\nOR\n\n' + raw_phi_fin if num != 0 else raw_phi_fin
        compute_log_likelihood_demos.cache_clear()
    return phi_all, complexity, raw_phi_all

def trajectorize(trajs_dict):
    """
    Change a dictionary with positive and negative examples used by AI-Interpret
    into a list of state-action pairs grouped in trajectories they came from.

    Parameters
    ----------
    trajs_dict : dict
        pos: [ (any ,any) ]
            Demonstrated (state, action) pairs
        neg: [ (any, any) ]
            (state, action) pairs for encountered states and non-optimal actions 
            according to the RL policy
        leng_pos: [ int ]
            Lengths of the consecutive demonstrations
        leng_neg: [ int ]
            Lengths of the negative examples for the consecutive demonstrations
        
    Returns
    -------
    trajs : [ [ (any, any) ] ]
        List of trajectories
    """
    trajs = []
    total_steps = 0
    for leng in trajs_dict['leng_pos']:
        subtraj = []
        for step in range(leng):
            sa_pair = trajs_dict['pos'][total_steps+step]
            subtraj.append(sa_pair)
        total_steps += len(subtraj)
        trajs.append(subtraj)
    return trajs

def simplify2(phi, until_cond):
    """
    Remove until_cond on not(until_cond) from a conjunction phi

    Parameters
    ----------
    phi : str
        Conjunction of predicates
    until_cond : str
        Predicates to be removed
        
    Returns
    -------
    phi_new : str
        Conjunction with the same ordering as phi but without until_cond or
        not(until_cond)
    """
    conjunctions = d2l(phi)[0]
    phi_new = ''
    for i, p in enumerate(conjunctions):
        if p != 'not(' + until_cond + ')' and until_cond != 'not(' + p + ')':
            if i == 0:
                phi_new += p
            else: 
                phi_new += ' and ' + p
    return phi_new


if __name__ == "__main__":
    cwd = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument('--formula', '-f',
                        type=str,
                        help="Name of the file with the DNF formula.")
    parser.add_argument('--preds', '-p',
                        type=str,
                        help="Name of the file with the matrix of evaluated predicates.")
    parser.add_argument('--demos', '-d',
                        type=str,
                        help="Name of the file with the demonstrations.")

    args = parser.parse_args()

    formula_path = cwd+'/interprets_formula/'
    folder_path = cwd+'/demos/'
    
    file_name = args.formula
    print("Retrieving the formula...")
    with open(formula_path + file_name, 'rb') as handle:
        dict_object = pickle.load(handle)
    formula = list(dict_object.keys())[0]
    print('Done')
    
    file_name = args.preds
    print("Retrieving the predicate matrix...")
    with open(folder_path + file_name, 'rb') as handle:
        pred_matrix = pickle.load(handle)
    print('Done')
        
    file_name = args.demos
    print("Retrieving the demos...")
    folder_path = cwd+'/demos/'
    with open(folder_path + file_name, 'rb') as handle:
        demos = pickle.load(handle)
    print('Done')

    #demos = join(demos, add_demos)
    trajs = trajectorize(demos)
    
    num_pos_demos = sum(pred_matrix[-1])
    pred_pos_matrix = pred_matrix[0][:num_pos_demos]
    
    allowed_preds = ['is_previous_observed_max(st, act)',
                     'is_positive_observed(st, act)',
                     'are_leaves_observed(st, act)',
                     'are_roots_observed(st, act)',
                     'is_previous_observed_positive(st, act)',
                     'is_previous_observed_min(st, act)',
                     'is_previous_observed_max_nonleaf(st, act)',
                     'is_previous_observed_max_leaf(st, act)',
                     'is_previous_observed_max_root(st, act)',
                     'is_previous_observed_max_level(st, act,  1 )',
                     'is_previous_observed_max_level(st, act,  2 )',
                     'is_previous_observed_max_level(st, act,  3 )',
                     'is_previous_observed_min_level(st, act,  1 )',
                     'is_previous_observed_min_level(st, act,  2 )',
                     'is_previous_observed_min_level(st, act,  3 )',
                     #'observed_count(st, act,  1 )',
                     #'observed_count(st, act,  2 )',
                     #'observed_count(st, act,  3 )',
                     #'observed_count(st, act,  4 )',
                     #'observed_count(st, act,  5 )',
                     #'observed_count(st, act,  6 )',
                     #'observed_count(st, act,  7 )',
                     #'observed_count(st, act,  8 )',
                     'termination_return(st, act,  -30 )',
                     'termination_return(st, act,  -25 )',
                     'termination_return(st, act,  -15 )',
                     'termination_return(st, act,  -10 )',
                     'termination_return(st, act,  0 )',
                     'termination_return(st, act,  10 )',
                     'termination_return(st, act,  15 )',
                     'termination_return(st, act,  25 )',
                     'termination_return(st, act,  30 )',
                     'is_max_in_branch(st, act)',
                     'are_branch_leaves_observed(st, act)']
                     #'is_previous_observed_sibling(st, act)',
                     #'is_previous_observed_parent(st, act)']
    combs = list(itertools.combinations(allowed_preds, 2))
    allowed_combs = ['(' + left + ' or ' + right + ')' for left, right in combs 
                     if (left[:10] != right[:10])]
    allowed_preds += allowed_combs

    redundant_types = ['all_']
    
    #preds = get_program_set(pred_pos_matrix.shape[1])[0]
    #allowed_preds += [pr.program for pr in preds if 'all_' in pr.program]

    print(formula)
    from IHP.modified_mouselab import TrialSequence
    from RL2DT.hyperparams import BRANCHING
    from RL2DT.strategy_demonstrations import reward as rew
    env = TrialSequence(1, [(BRANCHING, rew)]).trial_sequence[0]
    unobserved_nodes = env.get_unobserved_nodes()
    unobserved_node_labels = [node.label for node in unobserved_nodes]
    from new import take_action
    ac = -100
    while ac != 0:
        ac = take_action(formula, env, unobserved_node_labels)
        print(({n.label: n.value for n in env.observed_nodes}, ac))
        env.node_map[ac].observe()

    try:
        LTL_formula, c, raw_LTL_formula = DNF2LTL(
                                            phi=formula, 
                                            trajs=trajs, 
                                            predicate_matrix=pred_pos_matrix,
                                            allowed_predicates=allowed_preds,
                                            redundant_predicates=allowed_preds,
                                            redundant_predicate_types=redundant_types)
    except ConversionError:
        LTL_formula, c, raw_LTL_formula = DNF2LTL(phi=formula, 
                                                  trajs=trajs, 
                                                  predicate_matrix=pred_pos_matrix,
                                                  allowed_predicates=allowed_preds,
                                                  redundant_predicates=[])
    print(':\n\n{}\n\nComplexity: {}'.format(LTL_formula, c))
    print('\n\n\n{}'.format(raw_LTL_formula))
    
    ## lens = {k: len(list(set([i[-1][0] for i in dct[k]]))) for k in dct.keys()}
    ## all = sum(lens.values())
    ## proportions = {k : v/all for k,v, in lens.items()}
