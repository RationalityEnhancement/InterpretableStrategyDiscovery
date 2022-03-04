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

### THIS IS THE DSL THAT GENERATE THE RESULTS FROM RESULTS.XCL
### DO NOT CHANGE! PLP IS VERY DELICATE
## 22245 programs
## DSL.pkl

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
    This node is a root 
    """
    if action == 0:
        return False
    node = state.node_map[action]
    return node.is_root()

def is_child_observed(state, action):
    """ 
    The child of this node is observed 
    """
    if action == 0:
        return False
    node = state.node_map[action]
    obs = [n.observed for n in node.children]
    return any(obs)

def is_observed(state, action):
    """ 
    This node is observed 
    """
    if action == 0:
        return False
    node = state.node_map[action]
    return node.observed

def is_sibling_observed(state, action):
    """ 
    The sibling of this node is observed 
    """
    if action == 0:
        return False
    node = state.node_map[action]
    siblings = [n for n in node.parent.children if n.label != node.label]
    if siblings == []:
        return False
    return any([s.observed for s in siblings])

def is_child_positive_value(state, action):
    """ 
    One of this node's children has a positive value
    """
    if action == 0:
        return False
    node = state.node_map[action]
    obs_children = [n for n in node.children if n.observed]
    if obs_children == []:
        return False
    max_child_val = max([o.value for o in obs_children])
    return max_child_val > 0

def is_parent_positive_value(state, action):
    """ 
    This node's parent has a positive value
    """
    if action == 0:
        return False
    node = state.node_map[action]
    if not node.parent.observed and node.parent.label != 0:
        return False
    return node.parent.value > 0

def is_successor_max_leaf(state, action):
    """ 
    One of the successors of this node has a value of 48 (or any other maximum 
    value from the distribution)
    """
    if action == 0:
        return False
    node = state.node_map[action]
    return node.is_successor_highest_leaf()

def is_parent_observed(state, action):
    """ 
    The parent of this node is observed
    """
    if action == 0:
        return False
    node = state.node_map[action]
    if node.parent.label == 0:
        return True
    return node.is_parent_observed()

def is_parent_best(state, action):
    """ 
    The parent of this node is observed and has the highest value among all the 
    observed nodes
    """
    if action == 0:
        return False
    node = state.node_map[action]
    parent = node.parent
    if parent in state.observed_nodes:
        if parent.value >= max([node.value for node in state.observed_nodes]):
            return True
    elif parent.label == 0:
        max_val = max([node.value for node in state.observed_nodes])
        if state.observed_nodes == [] or (state.observed_nodes != [] and
                                          0 >= max_val):
            return True
    return False

def is_on_highest_expected_value_path(state, action):
    """ 
    This node lies on the branch that has the highest expected value
    """
    if action == 0:
        return False
    node = state.node_map[action]
    return node.on_most_promising_path()

def is_on_2highest_expected_value_path(state, action):
    """ 
    This node lies on the branch that has the second highest expected value
    """
    if action == 0:
        return False
    node = state.node_map[action]
    return node.on_second_promising_path()

def has_leaf_highest_level_value(state, action):
    """ 
    This node leads to leaf that has the maximum value of its level
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
    This node can be accessed through an observed root with the maximum 
    value of its level
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
    This node leads to leaf that has the minimum value of its level
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
    This node can be accessed through an observed root with the minimum 
    value of its level
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
    This node's child has the maximum value of its level
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
    This node's parent has the maximum value of its level
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
    This node's child has the minimum value of its level
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
    This node's parent has the minimum value of its level
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
    The level on which this node lies is equal to 'dep'
    """
    if action == 0:
        return False
    node = state.node_map[action]
    return node.get_depth_node() == dep

def value_max_successor(state, action, val):
    """ 
    The biggest value among the successors of this node is equal to 'val'
    """
    if action == 0:
        return False
    node = state.node_map[action]
    return node.get_max_successor_value() == val

def value_max_on_path(state, action, val):
    """ 
    This node lies on the branch on which a node with the maximum value lies as well
    """
    if action == 0:
        return False
    node = state.node_map[action]
    return node.best_largest_value_observed() == val

def value_max_child(state, action, val):
    """ 
    The biggest value among the children of this node is equal to 'val'
    """
    if action == 0:
        return False
    node = state.node_map[action]
    return node.get_max_immediate_successor() == val

def value_parent(state, action, val):
    """ 
    The value of the parent of this node is equal to 'val'
    """
    if action == 0:
        return False
    node = state.node_map[action]
    return node.get_parent_value() == val

def value_min_branch(state, action, val):
    """ 
    The minimum value of a node laying on the same branch as this node is 'val'
    """
    if action == 0:
        return False
    node = state.node_map[action]
    return node.soft_pruning() == val



################### GENERAL PREDICATES

def siblings_count(state, action, num):
    """ 
    The number of observed siblings of this node is >= num 
    """
    if action == 0:
        return False
    node = state.node_map[action]
    return node.get_observed_siblings_count() >= num

def depth_count(state, action, num):
    """ 
    The number of observed nodes at the same depth as this node is >= num 
    """
    if action == 0:
        return False
    node = state.node_map[action]
    return node.get_observed_same_depth_count() >= num

def parent_depth_count(state, action, num):
    """ 
    The number of observed nodes at this node parent's depth is >= num 
    """
    if action == 0:
        return False
    node = state.node_map[action].parent
    if node.label == 0:
        return 1 >= num
    return node.get_observed_same_depth_count() >= num

def ancestors_count(state, action, num):
    """ 
    The number of observed ancestors of this node is >= num 
    """
    if action == 0:
        return False
    node = state.node_map[action]
    return node.get_observed_ancestor_count() >= num

def successors_count(state, action, num):
    """ 
    The number of observed successors of this node is >= num 
    """
    if action == 0:
        return False
    node = state.node_map[action]
    return node.get_observed_successor_count() >= num

def children_count(state, action, num):
    """ 
    The number of observed children of this node is >= num 
    """
    if action == 0:
        return False
    node = state.node_map[action]
    return node.get_immediate_successor_count() >= num

def level_count(state, action, dep, num):
    """ 
    The number of all the observed nodes at depth 'dep' is >= num 
    """
    if action == 0:
        return False
    some_node = state.level_map[dep][0]
    return some_node.get_observed_same_depth_count() >= num

def min_observed_branch_count(state, action, num):
    """ 
    The number of observed branches passing through this node is >= num 
    """
    if action == 0:
        return False
    node = state.node_map[action]
    return node.count_observed_node_branch() >= num

def is_parent_depth_observed(state, action):
    """ 
    All the nodes one level above this node are observed 
    """
    o = [ob.label for ob in state.observed_nodes]
    if action == 0:
        return False
    node = state.node_map[action].parent
    if node.label == 0:
        return True
    else:
        nodes_at_depth = state.level_map[state.node_level_map[node.label]]
        for node in nodes_at_depth:
            if not node.observed:
                return False
    return True

def is_previous_observed_successor(state, action):
    """ 
    The previously observed node is one of the successors of this node
    """
    if action == 0:
        return False
    node = state.node_map[action]
    return node.is_previous_successor()

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

def is_previous_observed_leaf(state, action):
    """ 
    The previously observed node was a leaf
    """
    if state.previous_observed is None:
        return False
    return is_leaf(state, state.previous_observed.label)

def is_previous_observed_positive(state, action):
    """ 
    The previously observed node had a positive value
    """
    if state.previous_observed is None:
        return False
    return state.previous_observed.value > 0

def is_max_leaf_path_observed(state, action):
    """ 
    A path from the root to a leaf with a value 48 was observed
    """
    return state.is_max_path_observed()

def is_previous_observed_max(state, action):
    """ 
    The previously observed node uncovered a 48
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

def is_none_observed(state, action):
    """ 
    No node has been yet observed
    """
    return state.observed_nodes == []

def is_all_observed(state, action):
    """ 
    All the nodes have been already observed
    """
    return len(state.observed_nodes) == state.num_nodes -1

def is_path_observed(state, action):
    """ 
    A path from the root to a leaf traversing through this node has been observed
    """
    return state.single_path_completion_termination()

def is_depth_lower_previous(state, action):
    """ 
    This node is one or more levels above the previously observed node
    """
    if action == 0:
        return False
    node = state.node_map[action]
    if state.previous_observed:
        previous = state.previous_observed
        return node.depth > previous.depth
    else:
        return True

def are_roots_observed(state, action):
    """ 
    All the roots of the Mouselab MDP tree have been observed
    """
    return state.all_roots_observed()

def are_leaves_observed(state, action):
    """ 
    All the leaves of the Mouselab MDP tree have been observed
    """
    return state.all_leaf_nodes_observed()

def are_leaves_and_max_paths_observed(state, action):
    """ 
    All the leaves and all the paths to the node with a maximum value have been 
    observed
    """
    return state.are_max_paths_observed()

def are_positive_roots_and_successor_leaves_observed(state, action):
    return state.positive_root_leaves_termination()

def previous_observed_depth(state, action, dep):
    """ 
    The level on which the previously observed node lies is equal to 'dep'
    """
    if action == 0 or state.previous_observed is None:
        return False
    node = state.previous_observed
    return node.get_depth_node() == dep

def expected_value_max_path(state, action, ev):
    """ 
    The expected reward after stopping now and passing through this node
    is >= ev
    """
    if action == 0:
        return False
    node = state.node_map[action]
    return node.calculate_best_expected_value() >= ev

def termination_return(state, action, ev):
    """ 
    The expected reward after stopping now is >= ev
    """
    return max(state.get_path_expected_values()) >= ev



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
    This node' has a parent with an uncovered value that is higher than any 
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
    This node' has a parent with an uncovered value that is lower than any 
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
    is lower than any other observed successor-leaf value for the nodes 
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
    This node has a predecessor that is a root with an observed value that 
    is lower than any other observed successor-leaf value for the nodes 
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
    is lower than any other observed successor-leaf value for the nodes 
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
    This node has a predecessor that is a root with an observed value that 
    is lower than any other observed successor-leaf value for the nodes 
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

def has_leftmost(state, action, node_labels):
    if action == 0:
        return False
    for nl in node_labels:
        if nl < action:
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
    and inside that set it also satisifes 'predicate_possible' 
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

def create_grammar():
    grammar = {
        START : (#[['any_(st, act, ', PRED, ', ', PRED, ')'],
                  [['all_(st, act, lambda st, act: ', LIST_PRED_AMONG_DEPTH, ')'],
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
                   [1./18.] * 18 + [1./18., 1./18.]),
                   
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
                                 [0.2] * 5),

        LIST_PRED_DEPTH : ([[LDEPTH],
                            [LDEPTH, ' and ', LLEAF],
                            [LDEPTH, ' and ', LROOT],
                            [LDEPTH, ' and ', LPARENT],
                            [LDEPTH, ' and ', LCHILD],
                            [LDEPTH, ' and ', LDEPTH],
                            [LDEPTH, ' and ', LPRED]],
                            [1./7.] * 7),
        
        LIST_PRED_LEAF : ([[LLEAF],
                           [LLEAF, ' and ', LDEPTH],
                           [LLEAF, ' and ', LROOT],
                           [LLEAF, ' and ', LPARENT],
                           [LLEAF, ' and ', LCHILD],
                           [LLEAF, ' and ', LLEAF],
                           [LLEAF, ' and ', LPRED]],
                           [1./7.] * 7),
        
        LIST_PRED_ROOT : ([[LROOT],
                           [LROOT, ' and ', LDEPTH],
                           [LROOT, ' and ', LLEAF],
                           [LROOT, ' and ', LPARENT],
                           [LROOT, ' and ', LCHILD],
                           [LROOT, ' and ', LROOT],
                           [LROOT, ' and ', LPRED]],
                           [1./7.] * 7),
        
        LIST_PRED_PARENT : ([[LPARENT],
                             [LPARENT, ' and ', LDEPTH],
                             [LPARENT, ' and ', LLEAF],
                             [LPARENT, ' and ', LROOT],
                             [LPARENT, ' and ', LCHILD],
                             [LPARENT, ' and ', LPARENT],
                             [LPARENT, ' and ', LPRED]],
                             [1./7.] * 7),
        
        LIST_PRED_CHILD : ([[LCHILD],
                            [LCHILD, ' and ', LLEAF],
                            [LCHILD, ' and ', LROOT],
                            [LCHILD, ' and ', LPARENT],
                            [LCHILD, ' and ', LDEPTH],
                            [LCHILD, ' and ', LCHILD],
                            [LCHILD, ' and ', LPRED]],
                            [1./7.] * 7),
        
        LIST_PRED_PRED : ([[LPRED],
                           [LPRED, ' and ', LDEPTH],
                           [LPRED, ' and ', LLEAF],
                           [LPRED, ' and ', LPARENT],
                           [LPRED, ' and ', LCHILD],
                           [LPRED, ' and ', LPRED],
                           [LPRED, ' and ', LROOT]],
                           [1./7.] * 7),
        

        AMONG_DEPTH : ([['lambda st, act, lst: has_smallest_depth(st, act, lst)'],
                        ['lambda st, act, lst: has_largest_depth(st, act, lst)']],
                        [0.5, 0.5]),
        
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
                   [0.5, 0.5]),

        LLEAF : ([['is_leaf(st, act)'],
                  ['has_leaf_highest_level_value(st, act)'],
                  ['has_leaf_smallest_level_value(st, act)'],
                  ['not(is_leaf(st, act))'],
                  ['not(has_leaf_highest_level_value(st, act))'],
                  ['not(has_leaf_smallest_level_value(st, act))']],
                  [1./6.] * 6),
        
        LROOT : ([['is_root(st, act)'],
                  ['has_root_highest_level_value(st, act)'],
                  ['has_root_smallest_level_value(st, act)'],
                  ['not(has_root_highest_level_value(st, act))'],
                  ['not(has_root_smallest_level_value(st, act))'],
                  ['not(is_root(st, act))']],
                  [1./6.] * 6),
        
        LPARENT : ([['is_parent_positive_value(st, act)'],
                    ['has_parent_highest_level_value(st, act)'],
                    ['has_parent_smallest_level_value(st, act)'],
                    ['not(is_parent_positive_value(st, act))'],
                    ['not(has_parent_highest_level_value(st, act))'],
                    ['not(has_parent_smallest_level_value(st, act))']],
                    [1./6.] * 6),


        LCHILD : ([['is_child_positive_value(st, act)'],
                  ['has_child_highest_level_value(st, act)'],
                  ['has_child_smallest_level_value(st, act)'],
                  ['not(is_child_positive_value(st, act))'],
                  ['not(has_child_highest_level_value(st, act))'],
                  ['not(has_child_smallest_level_value(st, act))']],
                  [1./6.] * 6),
                  
                  
        LPRED : ([['is_on_highest_expected_value_path(st, act)'],
                  ['is_on_2highest_expected_value_path(st, act)'],
                  ['not(is_on_highest_expected_value_path(st, act))'],
                  ['not(is_on_2highest_expected_value_path(st, act))'],
                  ['not(is_observed(st, act))']],
                  [1./5.] * 5),

        GENERAL_PRED : ([['is_none_observed(st, act)'],
                         ['is_all_observed(st, act)'],
                         ['is_path_observed(st, act)'],
                         ['is_parent_depth_observed(st, act)'],
                         ['is_previous_observed_max(st, act)'],
                         ['is_positive_observed(st, act)'],
                         ['are_leaves_observed(st, act)'],
                         ['is_previous_observed_positive(st, act)'],
                         ['is_previous_observed_parent(st, act)'],
                         ['is_previous_observed_sibling(st, act)'],
                         ['is_previous_observed_max_level(st, act, ', DEP, ')'],
                         ['is_previous_observed_min_level(st, act, ', DEP, ')'],
                         ['previous_observed_depth(st, act, ', DEP, ')'],
                         ['siblings_count(st, act, ', NUM, ')'],
                         ['depth_count(st, act, ', NUM, ')'],
                         ['parent_depth_count(st, act, ', NUM, ')'],
                         ['ancestors_count(st, act, ', NUM, ')'],
                         ['successors_count(st, act, ', NUM, ')'],
                         ['children_count(st, act, ', NUM, ')'],
                         ['level_count(st, act, ', DEP, ', ', NUM, ')']],
                         [1./20.] * 20),

        NUM : ([[str(i)] for i in range(9)],
                [1./9.] * 9),

        DEP : ([['1'], ['2'], ['3']],
                [1./3., 1./3., 1./3.]),
    }
    return grammar
