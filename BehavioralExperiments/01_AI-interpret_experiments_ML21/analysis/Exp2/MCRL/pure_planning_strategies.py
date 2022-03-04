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

def observe_till_root(trial,node):
    present_node = node
    while(present_node.parent.label != 0):
        if present_node.parent.observed:
            break
        present_node.parent.observe()
        present_node = present_node.parent
    return

def observe_till_root_with_pruning(trial, node):
    present_node = node
    while(present_node.parent.label != 0):
        if present_node.parent.observed:
            return 0
        present_node.parent.observe()
        present_node = present_node.parent
        if present_node.value < 0 and not present_node.is_root():
            return 0
    return 1

def observe_randomly_till_root(trial,node):
    present_node = node
    nodes = []
    while(present_node.parent.label != 0):
        nodes.append(present_node.parent)
        present_node = present_node.parent
    shuffle(nodes)
    for node in nodes:
        if not node.observed:
            trial.node_map[node.label].observe()
    return

def get_nodes_till_root(trial,node):
    present_node = node
    nodes = []
    while(present_node.parent.label != 0):
        nodes.append(present_node.parent)
        present_node = present_node.parent
    return nodes

def observe_path_from_root_to_node(trial, node):
    present_node = node
    nodes = [present_node]
    while(present_node.parent.label != 0):
        nodes.append(present_node.parent)
        present_node = present_node.parent
    nodes = nodes[::-1]
    for node in nodes:
        if not node.observed:
            trial.node_map[node.label].observe()
    return

def observe_leaves_of_root(trial, root, satisficing_value = None):
    branch_nums = trial.reverse_branch_map[root.label]
    leaf_labels = []
    for branch_num in branch_nums:
        branch = trial.branch_map[branch_num]
        leaf_labels.append(branch[-1])
    shuffle(leaf_labels)
    for leaf in leaf_labels:
        node = trial.node_map[leaf]
        node.observe()
        if satisficing_value:
            if node.value >= satisficing_value:
                return 1
    return 0

def get_max_leaves(trial):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    leaf_values = []
    for node in leaf_nodes:
        node.observe()
        leaf_values.append(node.value)
    max_leaf_value = max(leaf_values)
    best_nodes = [node for node in leaf_nodes if node.value == max_leaf_value]
    return best_nodes

def observe_node_path_subtree(trial, node, random = True):
    observe_till_root(trial, node)
    present_node = node
    while(present_node.parent.label != 0):
        present_node = present_node.parent
    path_root = present_node
    if random:
        successors = path_root.get_successor_nodes()
        unobserved_successors = [node for node in successors if not node.observed]
        shuffle(unobserved_successors)
        for node in unobserved_successors:
            node.observe()
    else:
        def observe_successors(node):
            if not node.children:
                return
            successors = node.get_successor_nodes()
            unobserved_successors = [node for node in successors if not node.observed]
            for child in unobserved_successors:
                if not child.observed:
                    child.observe()
                observe_successors(child)
        observe_successors(path_root)
    return states

def compare_paths_satisficing(trial, best_nodes):
    # Currently looks at all observed values and stops clicking when
    # a definite best is found
    temp_pointers = [node for node in best_nodes]
    best_node_values = [node.value for node in best_nodes]
    max_node_value = max(best_node_values)
    max_nodes = [node for node in best_nodes if node.value == max_node_value]
    if len(max_nodes) == 1:
        return
    while(1):
        parent_pointers = []
        parent_values = []
        for i,p in enumerate(temp_pointers):
            if p.parent.label != 0:
                if not temp_pointers[i].parent.observed:
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

@strategy("BRFS")
def randomized_breadth_first_search(trial):
    max_depth = trial.max_depth
    max_value = trial.get_max_dist_value()
    for d in list(range(1,max_depth+1)):
        nodes = trial.level_map[d]
        shuffle(nodes)
        for node in nodes:
            node.observe()
            if node.value >= max_value:
                return [node.label for node in trial.observed_nodes] + [0]
    return [node.label for node in trial.observed_nodes] + [0]

@strategy("DFS")
def satisficing_dfs(trial):
    root_nodes = trial.node_map[0].children.copy()
    max_value = trial.get_max_dist_value()
    def dfs(node, trial):
        node.observe()
        if node.value >= max_value:
            return 1
        if not node.children:
            return 0
        else:
            shuffle(node.children)
            for child in node.children:
                res = dfs(child, trial)
                if res == 1:
                    return 1
        return None
    shuffle(root_nodes)
    for root_node in root_nodes:
        res = dfs(root_node, trial)
        if res == 1:
            break
    return [node.label for node in trial.observed_nodes] + [0]

@strategy("Immediate")
def check_all_roots(trial):
    """ Explores all root nodes and terminates """
    root_nodes = trial.node_map[0].children.copy()
    shuffle(root_nodes)
    for node in root_nodes:
        node.observe()
    return [node.label for node in trial.observed_nodes] + [0]

@strategy("Final")
def check_all_leaves(trial):
    """ Explores all leaf nodes and terminates """
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    max_value = trial.get_max_dist_value()
    for node in leaf_nodes:
        node.observe()
        #if node.value >= max_value:
        #    return [node.label for node in trial.observed_nodes] + [0]
    return [node.label for node in trial.observed_nodes] + [0]

@strategy("BEFS")
def satisficing_best_first_search_expected_value(trial):
    pq = PriorityQueue()
    pq.put((-0,0))
    rf = trial.reward_function
    max_value = trial.get_max_dist_value()
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
        best_child.observe()
        #if best_child.value >= max_value:
        #    break
        pq.put((-best_child.value, best_child.label))
    return [node.label for node in trial.observed_nodes] + [0]

@strategy("copycat")
def approximate_optimal(trial):
    max_value = trial.get_max_dist_value()
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    values = []
    for node in leaf_nodes:
        node.observe()
        values.append(node.value)
        if node.value >= max_value:
            return [node.label for node in trial.observed_nodes] + [0]
    max_observed_value = max(values)
    max_nodes = [node for node in leaf_nodes if node.value == max_observed_value]
    compare_paths_satisficing(trial, max_nodes)
    return [node.label for node in trial.observed_nodes] + [0]

@strategy("NO")
def goal_setting_equivalent_goals_level_by_level(trial):
    best_nodes = get_max_leaves(trial)
    temp_pointers = [node for node in best_nodes]
    while(1):
        parent_pointers = []
        shuffle(temp_pointers)
        for i,p in enumerate(temp_pointers):
            if p.parent.label != 0:
                if not temp_pointers[i].parent.observed:
                    temp_pointers[i].parent.observe()
                    parent_pointers.append(temp_pointers[i].parent)
        if len(parent_pointers) == 0:
            return [node.label for node in trial.observed_nodes] + [0]
        temp_pointers = parent_pointers

@strategy("NO2")
def goal_setting_equivalent_goals_leaf_to_root(trial):
    best_nodes = get_max_leaves(trial)
    shuffle(best_nodes)
    for node in best_nodes:
        observe_till_root(trial,node)
    return [node.label for node in trial.observed_nodes] + [0]

@strategy("NO3")
def goal_setting_equivalent_goals_random(trial):
    best_nodes = get_max_leaves(trial)
    shuffle(best_nodes)
    nodes_list = []
    for node in best_nodes:
        nodes_list += get_nodes_till_root(trial, node)
    nodes_list = list(set(nodes_list))
    shuffle(nodes_list)
    for node in nodes_list:
        node.observe()
    return [node.label for node in trial.observed_nodes] + [0]

@strategy("Inverse_NO2")
def goal_setting_equivalent_goals_root_to_leaf(trial):
    best_nodes = get_max_leaves(trial)
    shuffle(best_nodes)
    for node in best_nodes:
        states = observe_path_from_root_to_node(trial, node)
    return [node.label for node in trial.observed_nodes] + [0]

@strategy("NO4")
def goal_setting_backward_planning(trial):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    max_value = trial.get_max_dist_value()
    for node in leaf_nodes:
        node.observe()
        if node.value >= max_value:
            break
    return [node.label for node in trial.observed_nodes] + [0]
