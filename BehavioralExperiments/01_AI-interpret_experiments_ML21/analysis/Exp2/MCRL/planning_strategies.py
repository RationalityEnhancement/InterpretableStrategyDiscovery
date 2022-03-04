import gym
import copy
from random import shuffle, choice
from queue import PriorityQueue
from .modified_mouselab_new import TrialSequence, approx_max, approx_min
from numpy import argsort

strategy_dict = {}
def strategy(name):
    def wrapper(func):
        strategy_dict[name] = func
    return wrapper

def get_second_max_dist_value(trial):
    values = []
    for d in range(1, trial.max_depth + 1):
        values.append(approx_max(trial.reward_function(d), position = 1))
    return max(values)

def observe_till_root(trial,node,states):
    present_node = node
    while(present_node.parent.label != 0):
        if present_node.parent.observed:
            break
        trial_copy = copy.deepcopy(trial)
        states.append(trial_copy)
        present_node.parent.observe()
        present_node = present_node.parent
    return states

def observe_till_root_with_pruning(trial, node, states):
    present_node = node
    while(present_node.parent.label != 0):
        if present_node.parent.observed:
            return 0, states
        trial_copy = copy.deepcopy(trial)
        states.append(trial_copy)
        present_node.parent.observe()
        present_node = present_node.parent
        if present_node.value < 0 and not present_node.is_root():
            return 0, states
    return 1, states

def observe_randomly_till_root(trial,node,states):
    present_node = node
    nodes = []
    while(present_node.parent.label != 0):
        nodes.append(present_node.parent)
        present_node = present_node.parent
    shuffle(nodes)
    for node in nodes:
        if not node.observed:
            trial_copy = copy.deepcopy(trial)
            states.append(trial_copy)
            trial.node_map[node.label].observe()
    return states

def get_nodes_till_root(trial,node):
    present_node = node
    nodes = []
    while(present_node.parent.label != 0):
        nodes.append(present_node.parent)
        present_node = present_node.parent
    return nodes

def observe_path_from_root_to_node(trial, node, states):
    present_node = node
    nodes = [present_node]
    while(present_node.parent.label != 0):
        nodes.append(present_node.parent)
        present_node = present_node.parent
    nodes = nodes[::-1]
    for node in nodes:
        if not node.observed:
            trial_copy = copy.deepcopy(trial)
            states.append(trial_copy)
            trial.node_map[node.label].observe()
    return states

def observe_leaves_of_root(trial, root, states, satisficing_value = None):
    branch_nums = trial.reverse_branch_map[root.label]
    leaf_labels = []
    for branch_num in branch_nums:
        branch = trial.branch_map[branch_num]
        leaf_labels.append(branch[-1])
    shuffle(leaf_labels)
    for leaf in leaf_labels:
        node = trial.node_map[leaf]
        trial_copy = copy.deepcopy(trial)
        states.append(trial_copy)
        node.observe()
        if satisficing_value:
            if node.value >= satisficing_value:
                return 1, states
    return 0, states

def get_max_leaves(trial, states):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    leaf_values = []
    for node in leaf_nodes:
        trial_copy = copy.deepcopy(trial)
        states.append(trial_copy)
        node.observe()
        leaf_values.append(node.value)
    max_leaf_value = max(leaf_values)
    best_nodes = [node for node in leaf_nodes if node.value == max_leaf_value]
    return best_nodes, states

def observe_node_path_subtree(trial, node, states, random = True):
    states = observe_till_root(trial, node, states)
    present_node = node
    while(present_node.parent.label != 0):
        present_node = present_node.parent
    path_root = present_node
    if random:
        successors = path_root.get_successor_nodes()
        unobserved_successors = [node for node in successors if not node.observed]
        shuffle(unobserved_successors)
        for node in unobserved_successors:
            trial_copy = copy.deepcopy(trial)
            states.append(trial_copy)
            node.observe()
    else:
        def observe_successors(node):
            if not node.children:
                return
            successors = node.get_successor_nodes()
            unobserved_successors = [node for node in successors if not node.observed]
            for child in unobserved_successors:
                if not child.observed:
                    trial_copy = copy.deepcopy(trial)
                    states.append(trial_copy)
                    child.observe()
                observe_successors(child)
        observe_successors(path_root)
    return states

def compare_paths_satisficing(trial, best_nodes, states):
    # Currently looks at all observed values and stops clicking when
    # a definite best is found
    temp_pointers = [node for node in best_nodes]
    best_node_values = [node.value for node in best_nodes]
    max_node_value = max(best_node_values)
    max_nodes = [node for node in best_nodes if node.value == max_node_value]
    if len(max_nodes) == 1:
        return states
    while(1):
        parent_pointers = []
        parent_values = []
        for i,p in enumerate(temp_pointers):
            if p.parent.label != 0:
                if not temp_pointers[i].parent.observed:
                    trial_copy = copy.deepcopy(trial)
                    states.append(trial_copy)
                    temp_pointers[i].parent.observe()
                    parent_value = temp_pointers[i].parent.value
                    parent_pointers.append(temp_pointers[i].parent)
                    parent_values.append(parent_value)
        if parent_values:
            max_parent_value = max(parent_values)
            max_nodes = [node for node in parent_pointers if node.value == max_parent_value]
            if len(max_nodes) == 1:
                break
            temp_pointers = parent_pointers
        break
    return states

@strategy(1)
def goal_setting_random_path(trial):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    states = []
    for node in leaf_nodes:
        trial_copy = copy.deepcopy(trial)
        states.append(trial_copy)
        node.observe()
        if node.value > 0:
            states = observe_randomly_till_root(trial, node, states)
    for node in leaf_nodes:
        states = observe_randomly_till_root(trial, node, states)
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    return zip(states, [node.label for node in trial.observed_nodes] + [0])

@strategy(2)
def inverse_randomized_breadth_first_search(trial):
    max_depth = trial.max_depth
    states = []
    for d in list(range(1,max_depth+1))[::-1]:
        nodes = trial.level_map[d]
        shuffle(nodes)
        for node in nodes:
            trial_copy = copy.deepcopy(trial)
            states.append(trial_copy)
            node.observe()
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    return zip(states, [node.label for node in trial.observed_nodes] + [0])

@strategy(3)
def randomized_breadth_first_search(trial):
    max_depth = trial.max_depth
    max_value = trial.get_max_dist_value()
    states = []
    for d in list(range(1,max_depth+1)):
        nodes = trial.level_map[d]
        shuffle(nodes)
        for node in nodes:
            trial_copy = copy.deepcopy(trial)
            states.append(trial_copy)
            node.observe()
            if node.value >= max_value:
                trial_copy = copy.deepcopy(trial)
                states.append(trial_copy)
                return zip(states, [node.label for node in trial.observed_nodes] + [0])
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    return zip(states, [node.label for node in trial.observed_nodes] + [0])

@strategy(4)
def progressive_deepening(trial):
    leaf_nodes = trial.get_leaf_nodes()
    states = []
    shuffle(leaf_nodes)
    for node in leaf_nodes:
        states = observe_path_from_root_to_node(trial, node, states)
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    return zip(states, [node.label for node in trial.observed_nodes] + [0])

@strategy(5)
def best_first_search_expected_value(trial): # Pick according to best value starting from roots
    pq = PriorityQueue()
    pq.put((-0,0))
    rf = trial.reward_function
    states = []
    while not pq.empty():
        top = trial.node_map[pq.queue[0][1]] # pq.queue is not ordered according to priority. Only index 0 is right.
        best_child, best_child_value = None,-9999
        children = top.children.copy()
        shuffle(children)
        for child in children:
            if not child.observed:
                ev = rf(child.depth).expectation()
                if ev > best_child_value:
                    best_child = child
                    best_child_value = ev
        if best_child is None:
            pq.get()
            continue
        trial_copy = copy.deepcopy(trial)
        states.append(trial_copy)
        best_child.observe()
        pq.put((-best_child.value, best_child.label))
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    return zip(states, [node.label for node in trial.observed_nodes] + [0])

@strategy(6)
def ancestor_priority(trial):
    """ Choose nodes according to the priority 0.8*number of
    ancestor nodes + 0.2*number of successor nodes
    """
    unobserved_nodes = trial.unobserved_nodes.copy()
    unobserved_nodes.remove(trial.node_map[0])
    shuffle(unobserved_nodes)
    states = []
    while(len(unobserved_nodes) != 0):
        scores = []
        ancestor_scores = []
        for node in unobserved_nodes:
            ancestor_count = node.get_observed_ancestor_count()
            successor_count = node.get_observed_successor_count()
            score = 0.8*ancestor_count + 0.2*successor_count
            scores.append(score)
            ancestor_scores.append(ancestor_count)
        max_score = max(scores)
        max_indices = [i for i,s in enumerate(scores) if s == max_score]
        max_ancestor_scores = [ancestor_scores[i] for i in max_indices]
        max_max_ancestor_scores = max(max_ancestor_scores)
        max_total_nodes = [unobserved_nodes[max_indices[i]] for i,s in enumerate(max_ancestor_scores) if s == max_max_ancestor_scores]
        node = choice(max_total_nodes)
        trial_copy = copy.deepcopy(trial)
        states.append(trial_copy)
        node.observe()
        unobserved_nodes.remove(node)
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    return zip(states, [node.label for node in trial.observed_nodes] + [0])

@strategy(7)
def successor_priority(trial):
    """ Choose nodes according to the priority 0.8*number of
    successor nodes + 0.2*number of ancestor nodes
    """
    unobserved_nodes = trial.unobserved_nodes.copy()
    unobserved_nodes.remove(trial.node_map[0])
    shuffle(unobserved_nodes)
    states = []
    while(len(unobserved_nodes) != 0):
        scores = []
        successor_scores = []
        for node in unobserved_nodes:
            ancestor_count = node.get_observed_ancestor_count()
            successor_count = node.get_observed_successor_count()
            score = 0.8*successor_count + 0.2*ancestor_count
            scores.append(score)
            successor_scores.append(successor_count)
        max_score = max(scores)
        max_indices = [i for i,s in enumerate(scores) if s == max_score]
        max_successor_scores = [successor_scores[i] for i in max_indices]
        max_max_successor_scores = max(max_successor_scores)
        max_total_nodes = [unobserved_nodes[max_indices[i]] for i,s in enumerate(max_successor_scores) if s == max_max_successor_scores]
        node = choice(max_total_nodes)
        trial_copy = copy.deepcopy(trial)
        states.append(trial_copy)
        node.observe()
        unobserved_nodes.remove(node)
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    return zip(states, [node.label for node in trial.observed_nodes] + [0])

@strategy(8)
def backward_path_goal_setting(trial):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    states = []
    for node in leaf_nodes:
        trial_copy = copy.deepcopy(trial)
        states.append(trial_copy)
        node.observe()
        if node.value > 0:
            states = observe_till_root(trial, node, states)
    for node in leaf_nodes:
        states = observe_till_root(trial, node, states)
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    return zip(states, [node.label for node in trial.observed_nodes] + [0])

@strategy(9)
def backward_path_subtree_goal_setting(trial):
    # Explores randomly in same subtree
    # We consider the nearest subtree
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    states = []
    for node in leaf_nodes:
        if not node.observed:
            trial_copy = copy.deepcopy(trial)
            states.append(trial_copy)
            node.observe()
            if node.value > 0:
                states = observe_node_path_subtree(trial, node, states)
    for node in leaf_nodes:
        states = observe_till_root(trial, node, states)
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    return zip(states, [node.label for node in trial.observed_nodes] + [0])

@strategy(10)
def randomized_sibling_breadth_first_search(trial):
    max_depth = trial.max_depth
    states = []
    for d in list(range(1,max_depth+1)):
        nodes = trial.level_map[d]
        shuffle(nodes)
        for node in nodes:
            if not node.observed:
                trial_copy = copy.deepcopy(trial)
                states.append(trial_copy)
                node.observe()
                siblings = node.get_sibling_nodes()
                shuffle(siblings)
                for sibling in siblings:
                    trial_copy = copy.deepcopy(trial)
                    states.append(trial_copy)
                    sibling.observe()
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    return zip(states, [node.label for node in trial.observed_nodes] + [0])

@strategy(11)
def immediate_ancestor_priority(trial):
    unobserved_nodes = trial.unobserved_nodes.copy()
    unobserved_nodes.remove(trial.node_map[0])
    shuffle(unobserved_nodes)
    states = []
    while(len(unobserved_nodes) != 0):
        scores = []
        ancestor_scores = []
        for node in unobserved_nodes:
            immediate_ancestor_count = 1 if node.is_parent_observed() else 0
            immediate_successor_count = node.get_immediate_successor_count()
            ancestor_count = node.get_observed_ancestor_count()
            successor_count = node.get_observed_successor_count()
            score = 0.6*immediate_ancestor_count + 0.2*ancestor_count + (
                    0.15*immediate_successor_count + 0.05*successor_count)
            scores.append(score)
            ancestor_scores.append(immediate_ancestor_count)
        max_score = max(scores)
        max_indices = [i for i,s in enumerate(scores) if s == max_score]
        max_ancestor_scores = [ancestor_scores[i] for i in max_indices]
        max_max_ancestor_scores = max(max_ancestor_scores)
        max_total_nodes = [unobserved_nodes[max_indices[i]] for i,s in enumerate(max_ancestor_scores) if s == max_max_ancestor_scores]
        node = choice(max_total_nodes)
        trial_copy = copy.deepcopy(trial)
        states.append(trial_copy)
        node.observe()
        unobserved_nodes.remove(node)
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    return zip(states, [node.label for node in trial.observed_nodes] + [0])

@strategy(12)
def immediate_successor_priority(trial):
    unobserved_nodes = trial.unobserved_nodes.copy()
    unobserved_nodes.remove(trial.node_map[0])
    shuffle(unobserved_nodes)
    states = []
    while(len(unobserved_nodes) != 0):
        scores = []
        successor_scores = []
        for node in unobserved_nodes:
            immediate_ancestor_count = 1 if node.is_parent_observed() else 0
            immediate_successor_count = node.get_immediate_successor_count()
            ancestor_count = node.get_observed_ancestor_count()
            successor_count = node.get_observed_successor_count()
            score = 0.6*immediate_successor_count + 0.2*successor_count + (
                    0.15*immediate_ancestor_count + 0.05*ancestor_count)
            scores.append(score)
            successor_scores.append(immediate_successor_count)
        max_score = max(scores)
        max_indices = [i for i,s in enumerate(scores) if s == max_score]
        max_successor_scores = [successor_scores[i] for i in max_indices]
        max_max_successor_scores = max(max_successor_scores)
        max_total_nodes = [unobserved_nodes[max_indices[i]] for i,s in enumerate(max_successor_scores) if s == max_max_successor_scores]
        node = choice(max_total_nodes)
        trial_copy = copy.deepcopy(trial)
        states.append(trial_copy)
        node.observe()
        unobserved_nodes.remove(node)
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    return zip(states, [node.label for node in trial.observed_nodes] + [0])

@strategy(13)
def backward_path_immediate_goal_setting(trial):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    states=  []
    for node in leaf_nodes:
        if not node.observed:
            trial_copy = copy.deepcopy(trial)
            states.append(trial_copy)
            node.observe()
            if node.value > 0:
                states = observe_node_path_subtree(trial, node, states, random = False)
    for node in leaf_nodes:
        states = observe_till_root(trial, node, states)
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    return zip(states, [node.label for node in trial.observed_nodes] + [0])

@strategy(14)
def best_leaf_node_path(trial):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    states = []
    pq = PriorityQueue()
    for node in leaf_nodes:
        trial_copy = copy.deepcopy(trial)
        states.append(trial_copy)
        node.observe()
        pq.put((-node.value,node.label))
    while not pq.empty():
        _, node_num = pq.get()
        node = trial.node_map[node_num]
        states = observe_till_root(trial, node, states)
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    return zip(states, [node.label for node in trial.observed_nodes] + [0])

@strategy(15)
def non_terminating_approximate_optimal(trial):
    # Similar to 16 but doesn't terminate
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    states = []
    max_value = trial.get_max_dist_value()
    leaf_values = []
    for node in leaf_nodes:
        trial_copy = copy.deepcopy(trial)
        states.append(trial_copy)
        node.observe()
        leaf_values.append(node.value)
        if node.value >= max_value:
            states = observe_till_root(trial, node, states)
    indices = argsort(leaf_values)[::-1]
    for i in indices:
        states = observe_till_root(trial, leaf_nodes[i], states)
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    return zip(states, [node.label for node in trial.observed_nodes] + [0])
    

@strategy(16)
def goal_setting_backward_planning(trial):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    states = []
    max_value = trial.get_max_dist_value()
    for node in leaf_nodes:
        trial_copy = copy.deepcopy(trial)
        states.append(trial_copy)
        node.observe()
        if node.value >= max_value:
            states = observe_till_root(trial, node, states)
            break
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    return zip(states, [node.label for node in trial.observed_nodes] + [0])

@strategy(17)
def goal_setting_equivalent_goals_level_by_level(trial):
    states = []
    best_nodes, states = get_max_leaves(trial, states)
    # Check if this still holds in case of unequal path lengths
    temp_pointers = best_nodes
    while(1):
        parent_pointers = []
        shuffle(temp_pointers)
        for i,p in enumerate(temp_pointers):
            if p.parent.label != 0:
                if not temp_pointers[i].parent.observed:
                    trial_copy = copy.deepcopy(trial)
                    states.append(trial_copy)
                    temp_pointers[i].parent.observe()
                    parent_pointers.append(temp_pointers[i].parent)
        if len(parent_pointers) == 0:
            trial_copy = copy.deepcopy(trial)
            states.append(trial_copy)
            return zip(states, [node.label for node in trial.observed_nodes] + [0])
        temp_pointers = parent_pointers

@strategy(18)
def goal_setting_equivalent_goals_random(trial):
    states = []
    best_nodes, states = get_max_leaves(trial, states)
    shuffle(best_nodes)
    nodes_list = []
    for node in best_nodes:
        nodes_list += get_nodes_till_root(trial, node)
    nodes_list = list(set(nodes_list))
    shuffle(nodes_list)
    for node in nodes_list:
        trial_copy = copy.deepcopy(trial)
        states.append(trial_copy)
        node.observe()
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    return zip(states, [node.label for node in trial.observed_nodes] + [0])

@strategy(19)
def goal_setting_equivalent_goals_leaf_to_root(trial):
    states = []
    best_nodes, states = get_max_leaves(trial, states)
    shuffle(best_nodes)
    for node in best_nodes:
        states = observe_till_root(trial,node,states)
    return zip(states, [node.label for node in trial.observed_nodes] + [0])

@strategy(20)
def goal_setting_equivalent_goals_root_to_leaf(trial):
    states = []
    best_nodes, states = get_max_leaves(trial, states)
    shuffle(best_nodes)
    for node in best_nodes:
        states = observe_path_from_root_to_node(trial, node, states)
    return zip(states, [node.label for node in trial.observed_nodes] + [0])

@strategy(21)
def approximate_optimal(trial):
    max_value = trial.get_max_dist_value()
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    states = []
    values = []
    for node in leaf_nodes:
        trial_copy = copy.deepcopy(trial)
        states.append(trial_copy)
        node.observe()
        values.append(node.value)
        if node.value >= max_value:
            trial_copy = copy.deepcopy(trial)
            states.append(trial_copy)
            return zip(states, [node.label for node in trial.observed_nodes] + [0])
    max_observed_value = max(values)
    max_nodes = [node for node in leaf_nodes if node.value == max_observed_value]
    states = compare_paths_satisficing(trial, max_nodes, states)
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    return zip(states, [node.label for node in trial.observed_nodes] + [0])

@strategy(22)
def positive_forward_satisficing(trial):
    """ Terminate on finding positive root.
        If no positive root node is found,
        terminates after exploring all root nodes
    """
    root_nodes = trial.node_map[0].children.copy()
    shuffle(root_nodes)
    states = []
    for node in root_nodes:
        trial_copy = copy.deepcopy(trial)
        states.append(trial_copy)
        node.observe()
        if node.value > 0:
            break
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    return zip(states, [node.label for node in trial.observed_nodes] + [0])

@strategy(23)
def check_all_roots(trial):
    """ Explores all root nodes and terminates """
    root_nodes = trial.node_map[0].children.copy()
    shuffle(root_nodes)
    states = []
    for node in root_nodes:
        trial_copy = copy.deepcopy(trial)
        states.append(trial_copy)
        node.observe()
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    return zip(states, [node.label for node in trial.observed_nodes] + [0])

@strategy(24)
def check_all_leaves(trial):
    """ Explores all leaf nodes and terminates """
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    states = []
    max_value = trial.get_max_dist_value()
    for node in leaf_nodes:
        trial_copy = copy.deepcopy(trial)
        states.append(trial_copy)
        node.observe()
        #if node.value >= max_value:
        #    trial_copy = copy.deepcopy(trial)
        #    states.append(trial_copy)
        #    return zip(states, [node.label for node in trial.observed_nodes] + [0])
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    return zip(states, [node.label for node in trial.observed_nodes] + [0])

@strategy(25)
def satisficing_best_first_search_expected_value(trial):
    pq = PriorityQueue()
    pq.put((-0,0))
    rf = trial.reward_function
    max_value = trial.get_max_dist_value()
    states = []
    while not pq.empty():
        top = trial.node_map[pq.queue[0][1]] # pq.queue is not ordered according to priority. Only index 0 is right.
        best_child, best_child_value = None,-9999
        children = top.children.copy()
        shuffle(children)
        for child in children:
            if not child.observed:
                ev = rf(child.depth).expectation()
                if ev > best_child_value:
                    best_child = child
                    best_child_value = ev
        if best_child is None:
            pq.get()
            continue
        trial_copy = copy.deepcopy(trial)
        states.append(trial_copy)
        best_child.observe()
        #if best_child.value >= max_value:
        #    break
        pq.put((-best_child.value, best_child.label))
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    return zip(states, [node.label for node in trial.observed_nodes] +[0])

@strategy(26)
def backward_positive_satisficing(trial):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    states = []
    for node in leaf_nodes:
        trial_copy = copy.deepcopy(trial)
        states.append(trial_copy)
        node.observe()
        if node.value > 0:
            break
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    return zip(states, [node.label for node in trial.observed_nodes] + [0])

@strategy(27)
def one_final_outcome(trial):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    states = []
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    leaf_nodes[0].observe()
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    return zip(states, [node.label for node in trial.observed_nodes] + [0])

@strategy(28)
def one_immediate_outcome(trial):
    root_nodes = trial.node_map[0].children.copy()
    shuffle(root_nodes)
    states = []
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    root_nodes[0].observe()
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    return zip(states, [node.label for node in trial.observed_nodes] + [0])

@strategy(29)
def goal_setting_forward_planning(trial):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    states = []
    max_value = trial.get_max_dist_value() 
    for node in leaf_nodes:
        states.append(trial_copy)
        node.observe()
        if node.value >= max_value:
            states = observe_path_from_root_to_node(trial,node, states)
            break
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    return zip(states, [node.label for node in trial.observed_nodes] + [0])

@strategy(30)
def no_planning(trial):
    return (trial, [0])

@strategy(31)
def satisficing_dfs(trial):
    root_nodes = trial.node_map[0].children.copy()
    max_value = trial.get_max_dist_value()
    states = []
    def dfs(node, states, trial):
        trial_copy = copy.deepcopy(trial)
        states.append(trial_copy)
        node.observe()
        if node.value >= max_value:
            return 1, states
        if not node.children:
            return 0, states
        else:
            shuffle(node.children)
            for child in node.children:
                res, states = dfs(child, states, trial)
                if res == 1:
                    return 1, states
        return None, states
    shuffle(root_nodes)
    for root_node in root_nodes:
        res, states = dfs(root_node, states, trial)
        if res == 1:
            break
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    return zip(states, [node.label for node in trial.observed_nodes] +[0])

@strategy(32)
def positive_root_leaves(trial):
    root_nodes = trial.node_map[0].children.copy()
    shuffle(root_nodes)
    states = []
    for root in root_nodes:
        trial_copy = copy.deepcopy(trial)
        states.append(trial_copy)
        root.observe()
    for root in root_nodes:
        if root.value > 0:
            _, states = observe_leaves_of_root(trial, root, states)
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    return zip(states, [node.label for node in trial.observed_nodes] + [0])

@strategy(33)
def satisficing_positive_root_leaves(trial):
    root_nodes = trial.node_map[0].children.copy()
    shuffle(root_nodes)
    states = []
    max_value = trial.get_max_dist_value()
    for root in root_nodes:
        trial_copy = copy.deepcopy(trial)
        states.append(trial_copy)
        root.observe()
    for root in root_nodes:
        if root.value > 0:
            res, states = observe_leaves_of_root(trial, root, states,satisficing_value = max_value)
            if res == 1:
                break
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    return zip(states, [node.label for node in trial.observed_nodes] + [0])

@strategy(34)
def positive_satisficing_positive_root_leaves(trial):
    root_nodes = trial.node_map[0].children.copy()
    shuffle(root_nodes)
    states = []
    for root in root_nodes:
        trial_copy = copy.deepcopy(trial)
        states.append(trial_copy)
        root.observe()
        if root.value >= 0:
            _, states = observe_leaves_of_root(trial, root, states, satisficing_value = 0)
            break
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    return zip(states, [node.label for node in trial.observed_nodes] + [0])

@strategy(35)
def backward_planning_positive_outcomes(trial):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    states = []
    for leaf in leaf_nodes:
        trial_copy = copy.deepcopy(trial)
        states.append(trial_copy)
        leaf.observe()
        if leaf.value > 0:
            states = observe_till_root(trial,leaf,states)
            break
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    return zip(states, [node.label for node in trial.observed_nodes] + [0])

@strategy(36)
def best_first_search_observed_roots(trial):
    # Currently implemented as starting after looking at all roots
    pq = PriorityQueue()
    root_nodes = trial.node_map[0].children.copy()
    shuffle(root_nodes)
    states = []
    for node in root_nodes:
        trial_copy = copy.deepcopy(trial)
        states.append(trial_copy)
        node.observe()
        pq.put((-node.value, node.label))
    while(not pq.empty()):
        _, node_num = pq.get()
        node = trial.node_map[node_num]
        children = node.children.copy()
        if children:
            shuffle(children)
            for child in children:
                trial_copy = copy.deepcopy(trial)
                states.append(trial_copy)
                child.observe()
                pq.put((-child.value, child.label))
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    return zip(states, [node.label for node in trial.observed_nodes] + [0])

@strategy(37)
def satisficing_best_first_search_observed_roots(trial):
    # Currently implemented as starting after looking at all roots
    pq = PriorityQueue()
    root_nodes = trial.node_map[0].children.copy()
    shuffle(root_nodes)
    states = []
    max_value = trial.get_max_dist_value()
    for node in root_nodes:
        trial_copy = copy.deepcopy(trial)
        states.append(trial_copy)
        node.observe()
        if node.value >= max_value:
            trial_copy = copy.deepcopy(trial)
            states.append(trial_copy)
            return zip(states, [node.label for node in trial.observed_nodes] + [0])
        pq.put((-node.value, node.label))
    while(not pq.empty()):
        _, node_num = pq.get()
        node = trial.node_map[node_num]
        children = node.children.copy()
        if children:
            shuffle(children)
            for child in children:
                trial_copy = copy.deepcopy(trial)
                states.append(trial_copy)
                child.observe()
                if child.value >= max_value:
                    trial_copy = copy.deepcopy(trial)
                    states.append(trial_copy)
                    return zip(states, [node.label for node in trial.observed_nodes] + [0])
                pq.put((-child.value, child.label))
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    return zip(states, [node.label for node in trial.observed_nodes] + [0])

@strategy(38)
def loss_averse_backward_planning(trial):
    leaf_nodes = trial.get_leaf_nodes()
    max_value = trial.get_max_dist_value()
    shuffle(leaf_nodes)
    states = []
    for node in leaf_nodes:
        trial_copy = copy.deepcopy(trial)
        states.append(trial_copy)
        node.observe()
        if node.value >= max_value:
            status, states = observe_till_root_with_pruning(trial, node, states)
            if status == 1:
                break
    trial_copy = copy.deepcopy(trial)
    states.append(trial_copy)
    return zip(states, [node.label for node in trial.observed_nodes] + [0])

@strategy(39)
def random_planning(trial):
    nodes = list(trial.node_map.values())
    shuffle(nodes)
    states = []
    node_labels = [node.label for node in nodes]
    termination_index = node_labels.index(0)
    i = -1
    for node in nodes:
        i += 1
        trial_copy = copy.deepcopy(trial)
        states.append(trial_copy)
        if i == termination_index:
            return zip(states, [node.label for node in trial.observed_nodes] + [0])
        node.observe()

@strategy(40)
def copycat(trial, actions):
    states = []
    for action in actions:
        trial_copy = copy.deepcopy(trial)
        states.append(trial_copy)
        if action != 0:
            trial.node_map[action].observe()
        else:
            return zip(states, [node.label for node in trial.observed_nodes] + [0])


if __name__ == "__main__":
    from learning_utils import Participant
    p = Participant(exp_num = 'T1.1', pid = 1, 
    feature_criterion = 'normalize', excluded_trials=list(range(11)))
    trials = p.envs
    num_trials = len(trials)
    env = TrialSequence(
        num_trials,
        pipeline,
        ground_truth=trials
    )
    for strategy in range(1,40):
        trial = env.trial_sequence[0]
        trial.reset_observations()
        print(strategy)
        print(strategy_dict[strategy](trial))
