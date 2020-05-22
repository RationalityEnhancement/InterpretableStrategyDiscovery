import numpy as np

"""
In the ensuing predicates the two main arugments are 'state' and 'action'.

The predicates either posit something about the state or the action. 

The action corresponds to a particular node in the state, i.e. 

state = [ Distribution/int ]
action ~ state[action]

Hence, asking something about the action in the state, asks something about a 
particular node of the Mouselab MDP represented by 'state'.
"""

############### HELPER predicates

def is_successor_max_val(state, action):
    """ 
    One of the successors of this node is uncovered and has the maximum possible 
    value in the MDP
    """
    if action == 0:
        return False
    node = state.node_map[action]
    observed_successors = node.get_observed_successor_nodes()
    max_val = state.get_max_dist_value()
    return any([s.value == max_val for s in observed_successors])

def is_ancestor_max_val(state, action):
    """ 
    One of the ancestors of this node is uncovered and has the maximum possible 
    value in the MDP.
    """
    if action == 0:
        return False
    node = state.node_map[action]
    max_val = state.get_max_dist_value()
    par = node
    predecessors = []
    while par:
        par = par.parent
        if par: predecessors.append(par)
    predecessor_vals = [p.value if p.observed else 0 for p in predecessors]
    return any([p == max_val for p in predecessor_vals])

def is_on_highest_expected_value_path(state, action):
    """ 
    This node lies on a path that has the highest expected value.
    """
    if action == 0:
        return False
    node = state.node_map[action]
    return node.on_most_promising_path()


############### SIMPLE predicates

def is_leaf(state, action):
    """ 
    This node is a leaf 
    """
    if action == 0:
        return False
    node = state.node_map[action]
    return node.is_leaf()

def is_root(state, action):
    """ 
    This node is one of the nodes on level 1
    """
    if action == 0:
        return False
    node = state.node_map[action]
    return node.is_root()

def is_observed(state, action):
    """ 
    This node was already clicked and is observed
    """
    if action == 0:
        return False
    node = state.node_map[action]
    return node.observed

def is_max_in_branch(state, action):
    '''
    This node lies on a path with an uncovered global maximum value of the rewards
    '''
    if action == 0:
        return False
    succ = is_successor_max_val(state, action)
    predec = is_ancestor_max_val(state, action)
    max_val = state.get_max_dist_value()
    node = state.node_map[action]
    node_val = node.value if node.observed else 0 
    self_max = node_val == max_val
    return succ or predec or self_max

def is_2max_in_branch(state, action):
    '''
    This node lies on a path with 2 uncovered global maximum values of the rewards
    '''
    if action == 0:
        return False
    succ = is_successor_max_val(state, action)
    predec = is_ancestor_max_val(state, action)
    max_val = state.get_max_dist_value()
    node = state.node_map[action]
    node_val = node.value if node.observed else 0 
    self_max = node_val == max_val
    return (succ and predec) or (succ and self_max) or (predec and self_max)

def are_branch_leaves_observed(state, action):
    '''
    This node has successor leaves which are all observed
    '''
    if action == 0:
        return False
    node = state.node_map[action]
    if node.is_leaf():
        for n in node.get_sibling_nodes():
            if n not in state.observed_nodes:
                return False
        return node.observed
    successors = node.get_successor_nodes()
    successor_leaves = [n for n in successors if n.is_leaf()]
    observed_successor_leaves = [n for n in successor_leaves if n.observed]
    return len(successor_leaves) == len(observed_successor_leaves)

def has_leaf_highest_level_value(state, action):
    """ 
    This node leads to an uncovered leaf that has the maximum possible 
    value on its level
    """
    if action == 0:
        return False
    node = state.node_map[action]
    obs_leaves_node = [state.node_map[n] 
                       for branch_num in state.reverse_branch_map[node.label] 
                       for n in state.branch_map[branch_num] 
                       if state.node_map[n].observed and state.node_map[n].is_leaf()]
    
    if obs_leaves_node == [] or node.is_leaf():
        return False
    
    max_leaf_val_node = max([o.value for o in obs_leaves_node])
    leaf_depth = obs_leaves_node[0].depth
    max_leaf_val = max(state.reward_function(leaf_depth).vals)
    if max_leaf_val_node == max_leaf_val:
        return True
    return False

def has_root_highest_level_value(state, action):
    """ 
    This node can be accessed through an observed node on level 1 which has 
    the highest value on level 1
    """
    if action == 0:
        return False
    node = state.node_map[action]
    obs_roots = [state.node_map[n] 
                 for branch_num in state.reverse_branch_map[node.label] 
                 for n in state.branch_map[branch_num] 
                 if state.node_map[n].observed and state.node_map[n].is_root()]
    
    if obs_roots == [] or node.is_root():
        return False
    
    max_root_val_node = max([o.value for o in obs_roots])
    max_root_val = max(state.reward_function(1).vals)
    if max_root_val_node == max_root_val:
        return True
    return False

def has_leaf_smallest_level_value(state, action):
    """ 
    This node leads to leaf that has the minimum possible value of its level
    """
    if action == 0:
        return False
    node = state.node_map[action]
    obs_leaves_node = [state.node_map[n] 
                       for branch_num in state.reverse_branch_map[node.label] 
                       for n in state.branch_map[branch_num] 
                       if state.node_map[n].observed and state.node_map[n].is_leaf()]
    
    if obs_leaves_node == [] or node.is_leaf():
        return False
    
    min_leaf_val_node = min([o.value for o in obs_leaves_node])
    leaf_depth = obs_leaves_node[0].depth
    min_leaf_val = min(state.reward_function(leaf_depth).vals)
    if min_leaf_val_node == min_leaf_val:
        return True
    return False

def has_root_smallest_level_value(state, action):
    """ 
    This node can be accessed through an observed node on level 1 which has 
    the lowest value on level 1
    """
    if action == 0:
        return False
    node = state.node_map[action]
    obs_roots = [state.node_map[n] 
                 for branch_num in state.reverse_branch_map[node.label] 
                 for n in state.branch_map[branch_num] 
                 if state.node_map[n].observed and state.node_map[n].is_root()]
    
    if obs_roots == [] or node.is_root():
        return False
    
    min_root_val_node = min([o.value for o in obs_roots])
    min_root_val = min(state.reward_function(1).vals)
    if min_root_val_node == min_root_val:
        return True
    return False

def has_child_highest_level_value(state, action):
    """ 
    This node's child has the maximum possible value on its level
    """
    if action == 0:
        return False
    node = state.node_map[action]
    obs_children = [n for n in node.children if n.observed]

    if obs_children == []:
        return False

    max_child_val = max([o.value for o in obs_children])
    max_val = max(state.reward_function(node.depth+1).vals)
    if max_child_val == max_val:
        return True
    return False

def has_parent_highest_level_value(state, action):
    """ 
    This node's parent has the maximum possible value on its level
    """
    if action == 0:
        return False
    node = state.node_map[action]
    
    if not node.parent.observed:
        return False
    
    max_parent_val = node.parent.value
    max_val = max(state.reward_function(node.depth-1).vals)
    if max_parent_val == max_val:
        return True
    return False

def has_child_smallest_level_value(state, action):
    """ 
    This node's child has the minimum posible value of its level
    """
    if action == 0:
        return False
    node = state.node_map[action]
    obs_children = [n for n in node.children if n.observed]

    if obs_children == []:
        return False

    min_child_val = min([o.value for o in obs_children])
    min_val = min(state.reward_function(node.depth+1).vals)
    if min_child_val == min_val:
        return True
    return False

def has_parent_smallest_level_value(state, action):
    """ 
    This node's parent has the minimum possible value of its level
    """
    if action == 0:
        return False
    node = state.node_map[action]
    
    if not node.parent.observed:
        return False
    
    min_parent_val = node.parent.value
    min_val = min(state.reward_function(node.depth-1).vals)
    if min_parent_val == min_val:
        return True
    return False


def depth(state, action, dep):
    """ 
    This node lies on level 'dep'
    """
    if action == 0:
        return False
    node = state.node_map[action]
    return node.get_depth_node() == dep


################### GENERAL PREDICATES

def is_previous_observed_parent(state, action):
    """ 
    The previously observed node is the parent of this node
    """
    if action == 0 or state.previous_observed is None:
        return False
    node = state.node_map[action]
    return state.previous_observed.label == node.parent.label

def is_previous_observed_sibling(state, action):
    """ 
    The previously observed node is one of the siblings of this node
    """
    if action == 0 or state.previous_observed is None:
        return False
    node = state.node_map[action]
    sibling_labels = [n.label for n in node.parent.children 
                      if n.label != node.label]
    if sibling_labels == []:
        return False
    return state.previous_observed.label in sibling_labels

def is_previous_observed_positive(state, action):
    """ 
    The previously observed node had a positive value
    """
    if state.previous_observed is None:
        return False
    return state.previous_observed.value > 0

def is_previous_observed_max(state, action):
    """ 
    The previously observed node uncovered the maximum possible value in the MDP
    """
    return state.is_previous_max()

def is_previous_observed_max_level(state, action, dep):
    """ 
    The previously observed node uncovered a maximum possible value on that level
    """
    prev_obs_node = state.previous_observed
    if prev_obs_node == None:
        return False
    return prev_obs_node.value == max(state.reward_function(dep).vals)

def is_previous_observed_min_level(state, action, dep):
    """ 
    The previously observed node uncovered a minimum possible value on that level
    """
    prev_obs_node = state.previous_observed
    if prev_obs_node == None:
        return False
    return prev_obs_node.value == min(state.reward_function(dep).vals)

def is_positive_observed(state, action):
    """ 
    There is a node with a positive value observed
    """
    return state.is_positive_observed()

def are_roots_observed(state, action):
    """ 
    All nodes on level 1 have been observed
    """
    return state.all_roots_observed()

def are_leaves_observed(state, action):
    """ 
    All leaf nodes have been observed
    """
    return state.all_leaf_nodes_observed()

def termination_return(state, action, ev):
    """ 
    The expected reward after stopping now is >= ev
    """
    return max(state.get_path_expected_values()) >= ev

def is_previous_observed_max_leaf(state, action):
    """
    The previously observed node is a leaf and it uncovered the maximum 
    possible value in the MDP
    """
    prev_obs_node = state.previous_observed
    if prev_obs_node == None:
        return False
    was_leaf = is_leaf(state, prev_obs_node.label)
    return was_leaf and state.is_previous_max()

def is_previous_observed_max_root(state, action):
    """
    The previously observed node lies on level 1 and it uncovered the maximum 
    possible value in the MDP.
    """
    prev_obs_node = state.previous_observed
    if prev_obs_node == None:
        return False
    was_root = is_root(state, prev_obs_node.label)
    return was_root and state.is_previous_max()

def is_previous_observed_min(state, action):
    """
    Previously oberved node had the lowest value for the distribution
    """
    previous_observed = state.previous_observed
    if previous_observed:
        previous_value = previous_observed.value
        min_values = list(state.min_values_by_depth.values())
        min_value = min(min_values)
        return previous_value == min_value
    return False

def is_previous_observed_max_nonleaf(state, action):
    """
    Previously oberved node was a nonleaf with the highest value for 
    the distribution
    """
    num_levels = max(state.level_map.keys())
    for k in range(1, num_levels-1):
        if is_previous_observed_max_level(state, action, k):
            return True
    return False
        
def observed_count(state, action, num):
    """
    There are at least 'num' observed nodes
    """
    return len(state.observed_nodes) >= num


################### AMONG/ALL PREDICATES

"""
Predicates that reside inside the 'among' or 'all' accept 'node_labels' parameter. 

Each of the predicates necessarily checks if the node defined by 'action' is 
in fact inside the 'node_labels'.
"""

def has_largest_depth(state, action, node_labels):
    """ 
    This node is the deepest in the tree among the nodes from 'node_labels'
    """
    if action == 0:
        return False
    node = state.node_map[action]
    for nl in node_labels:
        deep = state.node_map[nl].depth
        if deep > node.depth:
            return False
    return True

def has_smallest_depth(state, action, node_labels):
    """ 
    This node is the shallowest in the tree among the nodes from 'node_labels'
    """
    if action == 0:
        return False
    node = state.node_map[action]
    for nl in node_labels:
        deep = state.node_map[nl].depth
        if deep < node.depth:
            return False
    return True

def has_parent_highest_value(state, action, node_labels):
    """ 
    This node has a parent with an observed value that is higher than any 
    other observed parent's value for the nodes from 'node_labels'
    """
    if action == 0 or action not in node_labels:
        return False
    node = state.node_map[action]
    if not node.parent.observed and node.parent.label != 0:
        return False
    for nl in node_labels:
        parent_nl = state.node_map[nl].parent
        if not parent_nl:
            continue
        if (parent_nl.observed or parent_nl.label == 0) and \
            parent_nl.value > node.parent.value:
            return False
    return True

def has_child_highest_value(state, action, node_labels):
    """ 
    This node has a child with an observed value that is higher than any 
    other observed child's value for the nodes from 'node_labels'
    """
    if action == 0 or action not in node_labels:
        return False
    node = state.node_map[action]
    obs_children = [n for n in node.children if n.observed]
    if obs_children == []:
        return False
    max_child_val = max([o.value for o in obs_children])
    for nl in node_labels:
        obs_children_nl = [n for n in state.node_map[nl].children if n.observed]
        if obs_children_nl == []:
            continue
        elif max([o.value for o in obs_children_nl]) > max_child_val:
            return False
        else:
            continue
    return True

def has_parent_smallest_value(state, action, node_labels):
    """ 
    This node has a parent with an observed value that is lower than any 
    other observed parent's value for the nodes from 'node_labels'
    """
    if action == 0  or action not in node_labels:
        return False
    node = state.node_map[action]
    if not node.parent.observed and node.parent.label != 0:
        return False
    for nl in node_labels:
        parent_nl = state.node_map[nl].parent
        if not parent_nl:
            continue
        if (parent_nl.observed or parent_nl.label == 0) and \
            parent_nl.value < node.parent.value:
            return False
    return True

def has_child_smallest_value(state, action, node_labels):
    """ 
    This node has a child with an observed value that is lower than any 
    other observed child's value for the nodes from 'node_labels'
    """
    if action == 0 or action not in node_labels:
        return False
    node = state.node_map[action]
    obs_children = [n for n in node.children if n.observed]
    if obs_children == []:
        return False
    min_child_val = min([o.value for o in obs_children])
    for nl in node_labels:
        obs_children_nl = [n for n in state.node_map[nl].children if n.observed]
        if obs_children_nl == []:
            continue
        elif max([o.value for o in obs_children_nl]) < min_child_val:
            return False
        else:
            continue
    return True

def has_leaf_highest_value(state, action, node_labels):
    """ 
    This node has a successor that is a leaf with an observed value that 
    is higher than any other observed successor-leaf's value for the nodes 
    from 'node_labels'
    """
    if action == 0 or action not in node_labels:
        return False
    node = state.node_map[action]
    obs_leaves_node = [state.node_map[n] 
                       for branch_num in state.reverse_branch_map[node.label] 
                       for n in state.branch_map[branch_num] 
                       if state.node_map[n].observed and \
                          state.node_map[n].is_leaf()]
    
    if obs_leaves_node == []:
        return False
    
    max_leaf_val_node = max([o.value for o in obs_leaves_node])
    for nl in node_labels:
        obs_leaves_nl = [state.node_map[n] 
                         for branch_num in state.reverse_branch_map[nl] 
                         for n in state.branch_map[branch_num] 
                         if state.node_map[n].observed and \
                            state.node_map[n].is_leaf()]
        if obs_leaves_nl == []:
            continue
        else:
            max_leaf_val_nl = max([o.value for o in obs_leaves_nl])
            if max_leaf_val_node < max_leaf_val_nl:
                return False
    return True

def has_root_highest_value(state, action, node_labels):
    """ 
    This node has an ancestor on level 1 with an observed value that 
    is higher than any other observed 1st-level ancestor's value for the nodes 
    from 'node_labels'
    """
    if action == 0 or action not in node_labels:
        return False
    node = state.node_map[action]
    obs_roots_node = [state.node_map[n] 
                      for branch_num in state.reverse_branch_map[node.label] 
                      for n in state.branch_map[branch_num] 
                      if state.node_map[n].observed and \
                         state.node_map[n].is_root()]
    
    if obs_roots_node == []:
        return False
    
    max_root_val_node = max([o.value for o in obs_roots_node])
    for nl in node_labels:
        obs_roots_nl = [state.node_map[n] 
                        for branch_num in state.reverse_branch_map[nl] 
                        for n in state.branch_map[branch_num] 
                        if state.node_map[n].observed and \
                           state.node_map[n].is_root()]
        if obs_roots_nl == []:
            continue
        else:
            max_root_val_nl = max([o.value for o in obs_roots_nl])
            if max_root_val_node < max_root_val_nl:
                return False
    return True

def has_leaf_smallest_value(state, action, node_labels):
    """ 
    This node has a successor that is a leaf with an observed value that 
    is lower than any other observed successor-leaf's value for the nodes 
    from 'node_labels'
    """
    if action == 0 or action not in node_labels:
        return False
    node = state.node_map[action]
    obs_leaves_node = [state.node_map[n] 
                       for branch_num in state.reverse_branch_map[node.label] 
                       for n in state.branch_map[branch_num] 
                       if state.node_map[n].observed and \
                          state.node_map[n].is_leaf()]
    
    if obs_leaves_node == []:
        return False
    
    min_leaf_val_node = min([o.value for o in obs_leaves_node])
    for nl in node_labels:
        obs_leaves_nl = [state.node_map[n] 
                         for branch_num in state.reverse_branch_map[nl] 
                         for n in state.branch_map[branch_num] 
                         if state.node_map[n].observed and \
                            state.node_map[n].is_leaf()]
        if obs_leaves_nl == []:
            continue
        else:
            min_leaf_val_nl = min([o.value for o in obs_leaves_nl])
            if min_leaf_val_node > min_leaf_val_nl:
                return False
    return True

def has_root_smallest_value(state, action, node_labels):
    """ 
    This node has an ancestor on level 1 with an observed value that 
    is lower than any other observed 1st-level ancestor's value for the nodes 
    from 'node_labels'
    """
    if action == 0 or action not in node_labels:
        return False
    node = state.node_map[action]
    obs_roots_node = [state.node_map[n] 
                      for branch_num in state.reverse_branch_map[node.label] 
                      for n in state.branch_map[branch_num] 
                      if state.node_map[n].observed and \
                         state.node_map[n].is_root()]
    
    if obs_roots_node == []:
        return False
    
    min_root_val_node = min([o.value for o in obs_roots_node])
    for nl in node_labels:
        obs_roots_nl = [state.node_map[n] 
                        for branch_num in state.reverse_branch_map[nl] 
                        for n in state.branch_map[branch_num] 
                        if state.node_map[n].observed and \
                           state.node_map[n].is_root()]
        if obs_roots_nl == []:
            continue
        else:
            min_root_val_nl = min([o.value for o in obs_roots_nl])
            if min_root_val_node > min_root_val_nl:
                return False
    return True

def has_most_branches(state, action, node_labels):
    '''
    This node belongs to the largest number of paths among the nodes in 
    node_labels
    '''
    if action == 0:
        return False
    num_branches = len(state.reverse_branch_map[action])
    for nl in node_labels:
        if len(state.reverse_branch_map[nl]) > num_branches:
            return False
    return True

def has_best_path(state, action, node_labels):
    '''
    This node lies on a path for which the sum of expected rewrds is the highest
    for the paths on which other nodes in node_labels lie
    '''
    if action == 0:
        return False
    expected_path_values = state.get_path_expected_values()
    node_paths = state.reverse_branch_map[action]
    best_path_node = max([expected_path_values[path] for path in node_paths])
    for nl in node_labels:
        nl_paths = state.reverse_branch_map[nl]
        best_path_nl = max([expected_path_values[path] for path in nl_paths])
        if best_path_nl > best_path_node:
            return False
    return True


################# COMPOSITIONAL PREDICATES

"""
Compositional predicates are predicates of the form PRED(A: B).

A is a SIMPLE predicate defined in the first part of this file.

B is an AMONG/ALL predicate (optional).
"""

def any_(state, action, predicate_necessary, predicate_possible):
    """ 
    Among all the nodes in the MDP that satisfy 'predicate_necessary' 
    at least one of them satisifes 'predicate_possible' 
    """
    necessary = []
    for i in range(state.num_nodes):
        satisfied = predicate_necessary(state, state.node_map[i].label)
        if satisfied:
            necessary.append(state.node_map[i].label)
    possible = [predicate_possible(state, n, necessary) for n in necessary]
    return any(possible)

def all_(state, action, predicate_necessary, predicate_possible):
    """ 
    All the nodes in the MDP that satisfy 'predicate_necessary' 
    also satisfy 'predicate_possible' 
    """
    necessary = []
    for i in range(state.num_nodes):
        satisfied = predicate_necessary(state, state.node_map[i].label)
        if satisfied:
            necessary.append(state.node_map[i].label)
    possible = [predicate_possible(state, n, necessary) for n in necessary]
    return all(possible)

def among(state, action, predicate_necessary, predicate_possible=None):
    """ 
    This node is among all the nodes in the MDP that satisfy 'predicate_necessary' 
    and inside that set it also satisfies 'predicate_possible' 
    """
    necessary = []
    for i in range(state.num_nodes):
        satisfied = predicate_necessary(state, state.node_map[i].label)
        if satisfied:
            necessary.append(state.node_map[i].label)
    if predicate_possible:
        if action in necessary and predicate_possible(state, action, necessary):
            return True
    else:
        if action in necessary:
            return True
    return False 


### Grammatical Prior
START                   = 0 
GENERAL_PRED            = 1
LDEPTH                  = 2
LCHILD                  = 3
LPARENT                 = 4
LLEAF                   = 5
LROOT                   = 6
LPRED                   = 7
AMONG_DEPTH             = 8
AMONG_CHILD             = 9
AMONG_PARENT            = 10
AMONG_LEAF              = 11
AMONG_ROOT              = 12
LIST_PRED_DEPTH         = 13 
LIST_PRED_CHILD         = 14
LIST_PRED_PARENT        = 15
LIST_PRED_LEAF          = 16
LIST_PRED_ROOT          = 17
LIST_PRED_PRED          = 18
LIST_PRED_AMONG_DEPTH   = 19 
LIST_PRED_AMONG_CHILD   = 20
LIST_PRED_AMONG_PARENT  = 21
LIST_PRED_AMONG_LEAF    = 22
LIST_PRED_AMONG_ROOT    = 23
LIST_PRED_AMONG_PRED    = 24
NUM                     = 25 
DEP                     = 26
RET                     = 27

def create_grammar():
    grammar = {
        START : ([['all_(st, act, lambda st, act: ', LIST_PRED_AMONG_DEPTH, ')'],
                  ['all_(st, act, lambda st, act: ', LIST_PRED_AMONG_LEAF, ')'],
                  ['all_(st, act, lambda st, act: ', LIST_PRED_AMONG_ROOT, ')'],
                  ['all_(st, act, lambda st, act: ', LIST_PRED_AMONG_PARENT, ')'],
                  ['all_(st, act, lambda st, act: ', LIST_PRED_AMONG_CHILD, ')'],
                  ['all_(st, act, lambda st, act: ', LIST_PRED_AMONG_PRED, ')'],
                  ['among(st, act, lambda st, act: ', LIST_PRED_AMONG_DEPTH, ')'],
                  ['among(st, act, lambda st, act: ', LIST_PRED_AMONG_LEAF, ')'],
                  ['among(st, act, lambda st, act: ', LIST_PRED_AMONG_ROOT, ')'],
                  ['among(st, act, lambda st, act: ', LIST_PRED_AMONG_PARENT, ')'],
                  ['among(st, act, lambda st, act: ', LIST_PRED_AMONG_CHILD, ')'],
                  ['among(st, act, lambda st, act: ', LIST_PRED_AMONG_PRED, ')'],
                  ['among(st, act, lambda st, act: ', LIST_PRED_DEPTH, ')'],
                  ['among(st, act, lambda st, act: ', LIST_PRED_LEAF, ')'],
                  ['among(st, act, lambda st, act: ', LIST_PRED_ROOT, ')'],
                  ['among(st, act, lambda st, act: ', LIST_PRED_PARENT, ')'],
                  ['among(st, act, lambda st, act: ', LIST_PRED_CHILD, ')'],
                  ['among(st, act, lambda st, act: ', LIST_PRED_PRED, ')'],
                  [LPRED],
                  [GENERAL_PRED]],
                  [1./20.] * 20),
                   
        LIST_PRED_AMONG_DEPTH : ([[LIST_PRED_DEPTH, ', ', AMONG_CHILD],
                                  [LIST_PRED_DEPTH, ', ', AMONG_PARENT],
                                  [LIST_PRED_DEPTH, ', ', AMONG_LEAF],
                                  [LIST_PRED_DEPTH, ', ', AMONG_ROOT]],
                                  [0.25] * 4),
        
        LIST_PRED_AMONG_LEAF : ([[LIST_PRED_LEAF, ', ', AMONG_CHILD],
                                 [LIST_PRED_LEAF, ', ', AMONG_PARENT],
                                 [LIST_PRED_LEAF, ', ', AMONG_ROOT],
                                 [LIST_PRED_LEAF, ', ', AMONG_DEPTH]],
                                 [0.25] * 4),
        
        LIST_PRED_AMONG_ROOT : ([[LIST_PRED_ROOT, ', ', AMONG_CHILD],
                                 [LIST_PRED_ROOT, ', ', AMONG_PARENT],
                                 [LIST_PRED_ROOT, ', ', AMONG_LEAF],
                                 [LIST_PRED_ROOT, ', ', AMONG_DEPTH]],
                                 [0.25] * 4),
        
        LIST_PRED_AMONG_PARENT : ([[LIST_PRED_PARENT, ', ', AMONG_CHILD],
                                   [LIST_PRED_PARENT, ', ', AMONG_ROOT],
                                   [LIST_PRED_PARENT, ', ', AMONG_LEAF],
                                   [LIST_PRED_PARENT, ', ', AMONG_DEPTH]],
                                   [0.25] * 4),
        
        LIST_PRED_AMONG_CHILD : ([[LIST_PRED_CHILD, ', ', AMONG_PARENT],
                                  [LIST_PRED_CHILD, ', ', AMONG_DEPTH],
                                  [LIST_PRED_CHILD, ', ', AMONG_LEAF],
                                  [LIST_PRED_CHILD, ', ', AMONG_ROOT]],
                                  [0.25] * 4),
        
        LIST_PRED_AMONG_PRED : ([[LIST_PRED_PRED, ', ', AMONG_PARENT],
                                 [LIST_PRED_PRED, ', ', AMONG_CHILD],
                                 [LIST_PRED_PRED, ', ', AMONG_DEPTH],
                                 [LIST_PRED_PRED, ', ', AMONG_LEAF],
                                 [LIST_PRED_PRED, ', ', AMONG_ROOT]],
                                 [1./5.] * 5),

        LIST_PRED_DEPTH : ([[LDEPTH],
                            [LDEPTH, ' and ', LLEAF],
                            [LDEPTH, ' and ', LROOT],
                            [LDEPTH, ' and ', LPARENT],
                            [LDEPTH, ' and ', LCHILD],
                            [LDEPTH, ' and ', LDEPTH],
                            [LDEPTH, ' and ', LPRED]],
                            [1./7.] * 7),
        
        LIST_PRED_LEAF : ([[LLEAF],
                           #[LLEAF, ' and ', LDEPTH],
                           [LLEAF, ' and ', LROOT],
                           [LLEAF, ' and ', LPARENT],
                           [LLEAF, ' and ', LCHILD],
                           [LLEAF, ' and ', LLEAF],
                           [LLEAF, ' and ', LPRED]],
                           [1./6.] * 6),
        
        LIST_PRED_ROOT : ([[LROOT],
                           #[LROOT, ' and ', LDEPTH],
                           #[LROOT, ' and ', LLEAF],
                           [LROOT, ' and ', LPARENT],
                           [LROOT, ' and ', LCHILD],
                           [LROOT, ' and ', LROOT],
                           [LROOT, ' and ', LPRED]],
                           [1./5.] * 5),
        
        LIST_PRED_PARENT : ([[LPARENT],
                             #[LPARENT, ' and ', LDEPTH],
                             #[LPARENT, ' and ', LLEAF],
                             #[LPARENT, ' and ', LROOT],
                             [LPARENT, ' and ', LCHILD],
                             [LPARENT, ' and ', LPARENT],
                             [LPARENT, ' and ', LPRED]],
                             [1./4.] * 4),
        
        LIST_PRED_CHILD : ([[LCHILD],
                            #[LCHILD, ' and ', LLEAF],
                            #[LCHILD, ' and ', LROOT],
                            #[LCHILD, ' and ', LPARENT],
                            #[LCHILD, ' and ', LDEPTH],
                            [LCHILD, ' and ', LCHILD],
                            [LCHILD, ' and ', LPRED]],
                            [1./3.] * 3),
        
        LIST_PRED_PRED : ([[LPRED],
                           #[LPRED, ' and ', LDEPTH],
                           #[LPRED, ' and ', LLEAF],
                           #[LPRED, ' and ', LPARENT],
                           #[LPRED, ' and ', LCHILD],
                           [LPRED, ' and ', LPRED]],
                           #[LPRED, ' and ', LROOT]],
                           [1./2.] * 2),
        

        AMONG_DEPTH : ([['lambda st, act, lst: has_smallest_depth(st, act, lst)'],
                        ['lambda st, act, lst: has_largest_depth(st, act, lst)'],
                        ['lambda st, act, lst: has_best_path(st, act, lst)'],
                        ['lambda st, act, lst: has_most_branches(st, act, lst)']],
                        [1./4.] * 4),
        
        AMONG_CHILD :  ([['lambda st, act, lst: has_child_highest_value(st, act, lst)'],
                         ['lambda st, act, lst: has_child_smallest_value(st, act, lst)']],
                         [0.5,0.5]),
        
        AMONG_PARENT : ([['lambda st, act, lst: has_parent_highest_value(st, act, lst)'],
                         ['lambda st, act, lst: has_parent_smallest_value(st, act, lst)']],
                         [0.5, 0.5]),
        
        AMONG_LEAF : ([['lambda st, act, lst: has_leaf_highest_value(st, act, lst)'],
                       ['lambda st, act, lst: has_leaf_smallest_value(st, act, lst)']],
                       [0.5, 0.5]),
                                 
        AMONG_ROOT : ([['lambda st, act, lst: has_root_highest_value(st, act, lst)'],
                       ['lambda st, act, lst: has_root_smallest_value(st, act, lst)']],
                       [0.5, 0.5]),

        LDEPTH : ([['depth(st, act, ', DEP, ')'],
                   ['not(depth(st, act, ', DEP, '))']],
                   [1./2., 1./2.]),

        LLEAF : ([['has_leaf_highest_level_value(st, act)'],
                  ['has_leaf_smallest_level_value(st, act)'],
                  ['not(has_leaf_highest_level_value(st, act))'],
                  ['not(has_leaf_smallest_level_value(st, act))']],
                  [1./4.] * 4),
        
        LROOT : ([['has_root_highest_level_value(st, act)'],
                  ['has_root_smallest_level_value(st, act)'],
                  ['not(has_root_highest_level_value(st, act))'],
                  ['not(has_root_smallest_level_value(st, act))']],
                  [1./4.] * 4),
        
        LPARENT : ([['has_parent_highest_level_value(st, act)'],
                    ['has_parent_smallest_level_value(st, act)'],
                    ['not(has_parent_highest_level_value(st, act))'],
                    ['not(has_parent_smallest_level_value(st, act))']],
                    [1./4.] * 4),


        LCHILD : ([['has_child_highest_level_value(st, act)'],
                   ['has_child_smallest_level_value(st, act)'],
                   ['not(has_child_highest_level_value(st, act))'],
                   ['not(has_child_smallest_level_value(st, act))']],
                   [1./4.] * 4),
                  
                  
        LPRED : ([['is_leaf(st, act)'],
                  ['is_root(st, act)'],
                  ['is_max_in_branch(st, act)'],
                  ['is_2max_in_branch(st, act)'],
                  ['are_branch_leaves_observed(st, act)'],
                  ['not(is_leaf(st, act))'],
                  ['not(is_root(st, act))'],
                  ['not(is_max_in_branch(st, act))'],
                  ['not(is_2max_in_branch(st, act))'],
                  ['not(are_branch_leaves_observed(st, act))'],
                  ['not(is_observed(st, act))']],
                  [1./11.] * 11),

        GENERAL_PRED : ([['is_previous_observed_max(st, act)'],
                         ['is_positive_observed(st, act)'],
                         ['are_leaves_observed(st, act)'],
                         ['are_roots_observed(st, act)'],
                         ['is_previous_observed_positive(st, act)'],
                         ['is_previous_observed_parent(st, act)'],
                         ['is_previous_observed_sibling(st, act)'],
                         ['is_previous_observed_min(st, act)'],
                         ['is_previous_observed_max_nonleaf(st, act)'],
                         ['is_previous_observed_max_leaf(st, act)'],
                         ['is_previous_observed_max_root(st, act)'],
                         ['is_previous_observed_max_level(st, act, ', DEP, ')'],
                         ['is_previous_observed_min_level(st, act, ', DEP, ')'],
                         ['observed_count(st, act, ', NUM, ')'],
                         ['termination_return(st, act, ', RET, ')']],
                         [1./15.] * 15),

        NUM : ([[str(i)] for i in range(9)],
                [1./9.] * 9),

        DEP : ([['1'], ['2'], ['3']],
                [1./3., 1./3., 1./3.]),
        
        RET : ([[str(i)] for i in [-30,-25,-15,-10,0,10,15,25,30]],
                [1./9.] * 9),
    }
    return grammar
