from sklearn import tree
from six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
from RL2DT.decision_tree_imitation_learning import get_program_set
from RL2DT.PLP.dt_utils import extract_plp_from_dt
from RL2DT.hyperparams import NUM_PROGRAMS

import pydotplus
import re
import numpy
import pickle
import os
import copy
import argparse
import re
import sys

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def unit_pred2expr(predicate_str, unit=False, first_conj=False,
                   second_conj=False, special=False, single=False):
    """
    Convert a string being a predicate to an expression.
    
    For *some of* the SIMPLE and GENERAL predicates from PLP.DSL
    """

    ## SIMPLE predicates
    if ' not' in predicate_str or 'not ' in predicate_str:
        real_pred = predicate_str[5:-1]
        not_ = 'not '
    elif ' not ' in predicate_str:
        real_pred = predicate_str[6:-1]
        not_ = 'not '
    elif 'not' in predicate_str:
        real_pred = predicate_str[4:-1]
        not_ = 'not '
    else:
        real_pred = predicate_str
        not_ = ''

    if re.match('\s{0,1}depth\(', real_pred):
        x = real_pred.split()[1]
        s= ''
        if not_ != '': 
            depth_not = ' any other'
            than = 'than '
        else:
            depth_not = ''
            than = ''
        if unit:
            return 'Is it on' + depth_not + ' level ' + than + x
        if second_conj:
            return ' and does it reside on' + depth_not + ' level ' + than + x
        if first_conj:
            return 'on' + depth_not + ' level ' + than + x
        if special:
            s = 's'
        ## 'that reside'+s+
        return 'on' +depth_not + ' level ' + than + x

    if 'on_highest' in real_pred:
        s = 's'
        if unit:
            return 'Is it '+not_+' on the best path'
        if not_ != '':
            not_ = 'does not '
            s = ''
        if first_conj:
            return 'a node that '+not_+'belong'+s+' to the best path'
        if second_conj:
            if not_ != '':
                return ' nor belongs to the best path'
            return ' and '+not_+'belong'+s+' to the best path'
        if special:
            return 'that '+not_+'belong'+s+' to the best path'
        return 'that '+not_+'belong'+s+' to the best path'
    if '2highest' in real_pred:
        s = 's'
        if unit:
            return 'Is it '+not_+' on the second-best path'
        if not_ != '':
            not_ = 'does not '
            s = ''
        if first_conj:
            return 'a node that '+not_+'belong'+s+' to the second-best path'
        if second_conj:
            return ' and '+not_+'belong'+s+' to the second-best path'
            if not_ != '':
                return ' nor belongs to the second-best path'
        if special:
            return 'that '+not_+'belong'+s+' to the second-best path'
        return 'that '+not_+'belong'+s+' to the second-best path'

    if 'is_leaf' in real_pred:
        if unit:
            return 'Is it'+not_+' a leaf'
        if not_ != '':
            not_ = 'non-'
        if first_conj:
            return 'a ' + not_ + 'leaf'
        if second_conj or special:
            return ' ' + not_ + 'leaf'
        return not_ + 'leaves'
    if 'is_root' in real_pred:
        if unit:
            return 'Is it'+not_+' a root'
        if not_ != '':
            not_ = 'non-'
        if first_conj:
            return 'a ' + not_ + 'root'
        if second_conj or special:
            return ' ' + not_ + 'root'
        return not_ + 'roots'
    if 'is_child' in real_pred:
        if not_ == '':
            which = 'positive'
        else:
            which = 'negative'
        if unit:
            return 'Does it\'s child have a '+which+' value'
        if first_conj:
            return 'a node that has a child with a '+which+' value'
        if second_conj:
            return ' and has a child with a '+which+' value'
        if special:
            return 'that has a child with a '+which+' value'
        return 'that have a child with a '+which+' value'
    if 'is_parent' in real_pred:
        if not_ == '':
            which = 'positive'
        else:
            which = 'negative'
        if unit:
                return 'Does it\'s parent have a '+which+' value'
        if first_conj:
            return 'a node that has a parent with a '+which+' value'
        if second_conj:
            return ' and has a parent with a '+which+' value'
        if special:
            return 'that has a parent with a '+which+' value'
        return 'that have a parent with a negative value'
    if 'highest' in real_pred or 'smallest' in real_pred:
        which = 'highest' if 'highest' in real_pred else 'lowest'
        if 'leaf' in real_pred:
            which = '48' if 'highest' in real_pred else '-48'
            if unit:
                return 'Does it lead to a leaf whose value is ' + which
            if not_ != '':
                finish = 'different from ' + which
            else:
                finish = which
            if first_conj:
                return 'a node that leads to a leaf whose value is ' + finish
            if second_conj:
                return ' and that leads to a leaf whose value is ' + finish
            if special:
                return 'that leads to a leaf whose value is ' + finish
            return 'that lead to a leaf whose value is ' + finish
        if 'root' in real_pred:
            which = '48' if 'highest' in real_pred else '-48'
            if unit:
                return 'Is it accessible through a root whose value is ' + which
            if not_ != '':
                finish = 'different from ' + which
            else:
                finish = which
            if first_conj:
                return 'a node accessible by a root whose value is ' + finish
            if second_conj:
                return ' accessible by a root whose value is ' + finish
                return 'that leads to a leaf whose value is ' + finish
            return 'accessible by a root whose value is ' + finish
        if 'child' in real_pred:
            if unit:
                return 'Does it have a child which has the ' + which \
                       + ' value on its level'
            if not_ != '':
                not_ = 'non-'
            else:
                not_ = 'the '
            if first_conj:
                return 'a node that has a child with ' + not_ + which \
                       + ' highest value on its level'
            if second_conj:
                return ' and has a child with ' + not_ + which \
                       + ' highest value on its level'
            if special:
                return 'that has a child with ' + not_ + which \
                       + ' highest value on its level'
            return 'that have a child with ' + not_ + which \
                   + ' highest value on its level'
        if 'parent' in real_pred:
            if unit:
                return 'Does it have a parent whose value is the ' \
                       + which + '  on its level'
            if not_ != '':
                not_ = 'non-'
            else:
                not_ = 'the '
            if first_conj:
                return 'a node that has a parent with ' + not_ + which \
                       + ' highest value on its level'
            if second_conj:
                return ' and has a parent with ' + not_ + which \
                       + ' highest value on its level'
            if special:
                return 'that has a parent with ' + not_ + which \
                       + ' highest value on its level'
            return 'that have a parent with the '+ not_ + which \
                   + ' value on its level'

    if 'is_observed' in real_pred:
        nodes = ''
        if not_ != '':
            not_ = 'un'
        if unit:
            return 'Is it '+not_+'observed'
        if single:
            nodes = ' nodes'
        return not_ + 'observed' + nodes

    ## GENERAL predicates
    if 'is_none' in real_pred:
        return 'Was nothing observed yet'
    if 'is_all' in real_pred:
        return 'Are all the nodes observed'
    if 'is_path' in real_pred:
        return 'Is there any path from root to leaf observed'
    if 'is_parent_depth' in real_pred:
        return 'Are all the nodes one level above observed'
    if 'is_previous_observed' in real_pred:
        e = 'Was the previously observed node '
        if 'positive' in real_pred:
            return 'Did the previous click uncover a positive value'
        if 'parent' in real_pred:
            return e + 'its parent'
        if 'sibling' in real_pred:
            return e + 'its sibling'
        if 'max_level' in real_pred:
            w = real_pred.split()
            num = w[1]
            return 'Did the previous click uncover the maximum value on level ' \
                    + num
        if 'min_level' in real_pred:
            w = real_pred.split()
            num = w[1]
            return 'Did the previous click uncover the minimum value on level ' \
                   + num
        if 'max' in real_pred:
            return 'Did the previous click uncover a 48'
    if 'is_positive' in real_pred:
        return 'Is a node with a positive value observed'
    if 'are_leaves' in real_pred:
        return 'Are all leaves observed'
    if 'previous_observed_depth' in real_pred:
        w = real_pred.split()
        num = w[1]
        return 'Was the previously observed node at level ' + num
    if 'count' in real_pred and 'level' not in real_pred:
        e = 'Is the number of all the observed '
        w = real_pred.split()
        num = w[1]
        if 'siblings' in real_pred:
            return e + 'siblings of this node greater or equal to ' + num
        if re.match('depth', real_pred):
            return e + 'nodes on the same level greater or equal to ' + num
        if 'parent_depth' in real_pred:
            return e + 'nodes one level above greater or equal to ' + num
        if 'ancestors' in real_pred:
            return e + 'nodes above greater or equal to ' + num
        if 'successors' in real_pred:
            return e + 'nodes below greater or equal to ' + num
        if 'children' in real_pred:
            return e + 'children of this node greater or equal to ' + num
    if 'level_count' in real_pred:
        w = real_pred.split()
        dep, num = w[1], w[3]
        return 'Is the number of all the observed nodes on level ' + dep \
               + ' greater or equal to ' + num

    return predicate_str


def pred2expr(predicate_str, unit=False, conjunction=False):
    """
    Join expressions two SIMPLE predicates or output an expression for one, if
    there is only one.
    """
    if ' or ' in predicate_str:
        expr = unit_pred2expr(p1, first_conj=True) \
                   + unit_pred2expr(p2,second_conj=True)
    if ' and ' in predicate_str:
        p1, p2 = predicate_str.split(' and ')
        if p1 == p2:
            print('A\n\n')
            expr = unit_pred2expr(p1, unit=True)
        elif 'is_observed' in p1: 
            print('B\n\n')
            if 'is_root' in p2 or 'is_leaf' in p2:
                expr = 'an ' + unit_pred2expr(p1, first_conj=True) \
                       + ' ' + unit_pred2expr(p2, special=True)
            else:
                if conjunction:
                    an = 'an '
                    node = ' node '
                else:
                    an = ''
                    node = ' nodes '
                expr = an + unit_pred2expr(p1, first_conj=True) + node \
                       + unit_pred2expr(p2, special=True)
        elif 'is_observed' in p2: 
            print('C\n\n')
            if 'is_root' in p1 or 'is_leaf' in p1:
                expr = 'an ' + unit_pred2expr(p2, first_conj=True) + ' ' \
                       + unit_pred2expr(p1, special=True)
            else:
                if conjunction:
                    an = 'an '
                    node = ' node '
                else:
                    an = ''
                    node = ' nodes '
                expr = an + unit_pred2expr(p2, first_conj=True) + node \
                       + unit_pred2expr(p1, special=True)
        elif 'is_root' in p1 or 'is_leaf' in p1:
            print('D\n\n')
            expr = unit_pred2expr(p1, first_conj=True) + ' ' \
                   + unit_pred2expr(p2, special=True)
        elif 'is_root' in p2 or 'is_leaf' in p2:
            print('E\n\n')
            expr = unit_pred2expr(p2, first_conj=True) + ' ' \
                   + unit_pred2expr(p1, special=True)
        elif not conjunction:
            print('F\n\n')
            expr = 'nodes ' + unit_pred2expr(p1) + ' and ' + unit_pred2expr(p2)
        else:
            print('G\n\n')
            expr = unit_pred2expr(p1, first_conj=True) \
                   + unit_pred2expr(p2,second_conj=True)
    else:
        expr = unit_pred2expr(predicate_str, unit=unit, single=True)
    return expr

def among_pred2expr(predicate_str, all_=False):
    """
    Convert a string being a predicate to an expression.
    
    For AMONG/ALL predicates from PLP.DSL
    """
    if 'has_smallest' in predicate_str:
        if all_:
            return 'are on the same level'
        return 'on the lowest level'
    if 'has_largest' in predicate_str:
        if all_:
            return 'are on the same level'
        return 'on the highest level'

    if 'has_best_path' in predicate_str:
       return "lie on a best path" 
    
    expr = predicate_str
    if 'has_child_highest' in predicate_str:
        expr = "has a child with the highest value"
        if all_:
            return 'have a child with the same value'
    if 'has_child_smallest' in predicate_str:
        expr = "has a child with the lowest value"
        if all_:
            return 'have a child with the same value'
    if 'has_parent_highest' in predicate_str:
        expr = "has a parent with the highest value"
        if all_:
            return 'have a parent with the same value'
    if 'has_parent_smallest' in predicate_str:
        expr = "has a parent with the lowest value"
        if all_:
            return 'have a parent with the same value'
    if 'has_leaf_highest' in predicate_str:
        expr = "leads to a leaf with the highest value"
        if all_:
            return 'lead to a leaf with the same value'
    if 'has_leaf_smallest' in predicate_str:
        expr = "leads to a leaf with the lowest value"
        if all_:
            return 'lead to a leaf with the same value'
    if 'has_root_highest' in predicate_str:
        expr = "can be accessed by a root with the highest value"
        if all_:
            return 'are accessible by a root with the same value'
    if 'has_root_smallest' in predicate_str:
        expr = "can be accessed by a root with the lowest value"
        if all_:
            return 'are accessible by a root with the same value'
 
    return expr
    

def logic2expr(pred):
    """
    Convert a string being a predicate to an expression.
    
    For COMPSITIONAL predicates from PLP.DSL
    """
    if re.match('among\(among', pred):
        amongs = re.search(':.+ \)', pred).group(0)[2:-2]
        among_pred_close = amongs.split(' : ')[0][:-1]
        among_pred_far = amongs.split(' : ')[1]
        print(pred)
        list_preds = re.search('.+[a-z]{1}\)', pred[11:]).group(0).split(' : ')[0][1:]
        expr = 'Consider all the nodes which ' + pred2expr(list_preds) \
               + '. Is the node in that set and ' + among_pred2expr(among_pred_close)\
               + '. Now consider the subset of nodes which satisfy the above.' \
               + 'The node, ' + among_pred2expr(among_pred_far)
                
    elif re.match('among', pred):
        if re.search(':', pred):
            among_pred = re.search(':.+ \)', pred).group(0)[2:-2]
            print(among_pred)
            list_preds = re.search('.+(?= :)', pred[6:]).group(0)
            print(list_preds)
            if 'has_smallest' in among_pred or 'has_largest' in among_pred:
                expr = 'Is it ' + among_pred2expr(among_pred) + ' among ' \
                        + pred2expr(list_preds)
            else:
                expr = 'Is it that among nodes which are' \
                       + pred2expr(list_preds) + ' this node ' \
                       + among_pred2expr(among_pred)
                
        else:
            list_preds = pred[6:-2]
            expr = 'Is it ' + pred2expr(list_preds, conjunction=True)
            
    elif re.match('all_', pred):
        among_pred = re.search(':.+ \)', pred).group(0)[2:-2]
        list_preds = re.search('.+(?= :)', pred[5:]).group(0)
        expr = 'Do all the nodes that ' + pred2expr(list_preds) + ' ' \
               + among_pred2expr(among_pred, all_=True)

    else:
        expr = unit_pred2expr(pred, unit=True)

    print(pred)
    expr += '?'
    print(expr)
    return expr

def prettify(pred):
    """
    Get rid of arguments, unimportant parentheses, etc.
    """
    if pred[:12] == "LOOP FROM ((":
        pred = pred[:10] + pred[12:-2]
    if pred[:2] == "((":
        pred = pred[2:-2]
    new_pred = re.sub('st, act, lambda st, act:  ', '', pred)
    new_pred = re.sub('\(st, act\)', '', new_pred)
    new_pred = re.sub('\(st, act, lst\)', '', new_pred)
    new_pred = re.sub('\(st, act ', '', new_pred)
    new_pred = re.sub(',  lambda st, act, lst', '', new_pred)
    new_pred = re.sub('st, act, ', '', new_pred)
    new_pred = re.sub('\):', ') :', new_pred)
    new_pred = re.sub(' new.*', '', new_pred)
    new_pred = re.sub('  ', ' ', new_pred)
    new_pred = re.sub('\) \)', '))', new_pred)
    new_pred = re.sub('not \(', 'not(', new_pred)
    new_pred = re.sub(r'(\w+)( )(\))', r'\1\3', new_pred)
    return new_pred

def check_for_additional_nodes(preds_in_dnfs, set_preds):
    """
    Check if the list of dnfs contains predicates that are the same but the
    overall form of a dnf is different (and thus, they should be given separate
    nodes in the tree).
    """
    new_set_preds = copy.deepcopy(set_preds)
    new_preds_in_dnfs = copy.deepcopy(preds_in_dnfs)
    prev_preds = {}
    for pred in set_preds:
        dnf_count = -1
        for dnf in preds_in_dnfs:
            dnf_count += 1
            sub_count = -1
            for sub_pred in dnf:
                sub_count += 1
                if (pred == sub_pred or pred == sub_pred[4:-1]) and \
                    pred not in prev_preds.keys():
                    ind = dnf.index(sub_pred)
                    prev_preds[pred] = [dnf[:ind]]
                elif (pred == sub_pred or pred == sub_pred[4:-1]) and \
                    pred in prev_preds.keys():
                    ind = dnf.index(sub_pred)
                    new_name = ' new' * len(prev_preds[pred])
                    if dnf[:ind] not in prev_preds[pred]:
                        if 'not' in sub_pred:
                            new_set_preds.append(sub_pred[4:-1] + new_name)
                        else:
                            new_set_preds.append(sub_pred + new_name)
                        new_preds_in_dnfs[dnf_count][sub_count] += new_name
                        prev_preds[pred].append(dnf[:ind])
                    elif prev_preds[pred].index(dnf[:ind]) != 0:
                        which = prev_preds[pred].index(dnf[:ind])
                        new_name = ' new' * which
                        if 'not' in sub_pred:
                            not_pred = new_preds_in_dnfs[dnf_count][sub_count]
                            change = 'not(' + not_pred[4:-1] + new_name + ')'
                            new_preds_in_dnfs[dnf_count][sub_count] = change
                        else:
                            new_preds_in_dnfs[dnf_count][sub_count] = new_name
                        
    #print(new_preds_in_dnfs)
    #print(new_set_preds)
    return new_preds_in_dnfs, new_set_preds

def dnf2conj_list(program_string, paran=True):
    """
    Transform a string encoding a dnf into a list of conjunctions.
    
    DNFs are themselves lists of strings for the predicates.
    """
    if program_string == 'False': return [['False']]
    dnfs = program_string.split(' or ')
    if paran: dnfs = [c[2:-2] for c in dnfs] ##remove (( ))
    dnfs = [re.sub('not ', 'not', c) for c in dnfs]
    preds_in_dnfs = [c.split(' and ') for c in dnfs]
    for dnf in preds_in_dnfs: ## correcting for spacing and among(A)
        c = -1
        for pred in dnf:
            c += 1
            if 'not (not' in pred:
                dnf[c] = pred[9:-2]
            if 'not(not' in pred:
                dnf[c] = pred[8:-2]
            if re.match('among', pred):
                open_ = sum([el == '(' for el in pred])
                close_ = sum([el == ')' for el in pred])      
                if not re.search(':', pred) and not re.search('and', pred) and open_ == close_:
                        dnf[c] = pred[6:]         
    for dnf in preds_in_dnfs: ##correcting for among(A and B: C)
        c = -1
        real_preds = []
        to_remove = []
        for almost_pred in dnf:
            c += 1
            open_ = sum([el == '(' for el in almost_pred])
            close_ = sum([el == ')' for el in almost_pred])
            if open_ > close_:
                real_pred = almost_pred + ' and ' + dnf[c+1]
                real_preds.append((c,real_pred))
                to_remove.append(c+1)
        for pair in real_preds:
            dnf[pair[0]] = pair[1]
        num_removed = 0
        for i in to_remove:
            dnf.remove(dnf[i-num_removed])
            num_removed += 1
    return preds_in_dnfs

def get_used_preds(preds_in_dnfs):
    """
    Extract all the unique strings from a list of dnfs.
    """
    set_preds = []
    def not_in(el, lis):
        return not(el in lis)
    for dnf in preds_in_dnfs:
        set_preds += [pred[4:-1] \
            if (re.match('not', pred) and not_in(pred[4:-1], set_preds)) \
            else pred if (not_in(pred, set_preds) and not_in(pred[4:-1], set_preds)) \
            else None for pred in dnf]
    set_preds = [pred for pred in set_preds if pred is not None]
    return set_preds

def set_digraph_ID_and_create_visited_nodes_dict(new_string, set_preds, expr=True):
    """
    Create IDs for nodes in the digraph file.
    
    Output a dictionary encoding what ID each predicate from the dnf should get.
    
    Each predicate appears in the digraph in its positive form.
    
    0 means a predicate not seen (defult)
    1 means predicate added as a positive predicate
    2 means predicate should be added as a negative one but waits for a
    positive occurrence
    4 means predicate added as a negative leaf
    3 means predicate added in both ways
    
    The ordering of adding tags determines the direction of the arrows in the
    graphs.
    """
    ids = {}
    ordering = {}
    for i in range(len(set_preds)):
        pretty_pred = prettify(set_preds[i])
        if expr:
            pretty_pred = logic2expr(pretty_pred)
        params = ' [fillcolor=white, label="' + pretty_pred + '"];\n'
        new_string += str(i) + params
        ids[set_preds[i]] = str(i)
        ids['not('+set_preds[i]+')'] = str(i)
        ordering[set_preds[i]] = 0
    return ids, ordering, new_string

def add_non_leaf_positive_pred_to_digraph(pred, dnf, c, new_string, set_preds, \
                                                        ordering, ids, saved):
    """
    Update the digraph file with information on a non-leaf node for a positive
    predicate.
    """
    if ordering[pred] == 0:
        params = '  [headlabel="Yes", labelangle=45, labeldistance="2.5"];\n'
        new_string += ids[pred] + ' -> ' + ids[dnf[c+1]] + params
        ordering[pred] = 1
        
    elif ordering[pred] in [1,3]:
        pass
    
    else:
        num, tag = ordering[pred]
        assert num in [2,4], 'The ordering should be 2/4 but is {}'.format(num)
        ordering[pred] = 3
        params = '  [headlabel="Yes", labelangle=45, labeldistance="2.5"];\n'
        new_string += ids[pred] + '-> ' + ids[dnf[c+1]] + params
        params = '  [headlabel="No", labelangle=-45, labeldistance="2.5"];\n'
        new_string += ids[pred] + '-> ' + tag + params
                      
    return new_string, ordering, saved

def add_leaf_positive_pred_to_digraph(pred, new_string, set_preds, \
                                            ordering, ids, saved):
    """
    Update the digraph file with information on leaf node for a positive
    predicate.
    """
    print('Positive')
    print(pred)
    print(ordering[pred])
    print('\n')
    saved += 1
    params = ' [fillcolor=forestgreen, fontcolor=green, label="CLICK IT"];\n'
    new_string += str(len(set_preds)+saved) + params
    if ordering[pred] == 0:
        ordering[pred] = 1
        params = '  [headlabel="Yes", labelangle=45, labeldistance="2.5"];\n'
        new_string += ids[pred] + '-> ' + str(len(set_preds)+saved) + params
        
    elif ordering[pred] in [1,3]:
        pass
    
    else:
        num, tag = ordering[pred]
        assert num in [2,4], 'The ordering should be 2/4 but is {}'.format(num)
        if num == 2:
            params = '  [headlabel="Yes", labelangle=45, labeldistance="2.5"];\n'
            new_string += ids[pred] + '-> ' + str(len(set_preds)+saved) + params
            params = '  [headlabel="No", labelangle=-45, labeldistance="2.5"];\n'
            new_string += ids[pred] + '-> ' + tag + params
        if num == 4:
            raise Exception('That is stupid, predicate {} if negated leads to\
                            True and now you want to send it to True when \
                            non-negated...'.format(pred))
        
    return new_string, ordering, saved

def add_positive_pred_to_digraph(pred, dnf, c, num_preds, new_string, \
                                    set_preds, ordering, ids, saved):
    """
    Update the digraph file with information on a node for a positive predicate.
    """
    if c+1 < num_preds:
        res = add_non_leaf_positive_pred_to_digraph(pred, 
                                                    dnf, 
                                                    c, 
                                                    new_string, 
                                                    set_preds, 
                                                    ordering, 
                                                    ids, 
                                                    saved)
        
    else:
        res = add_leaf_positive_pred_to_digraph(pred, 
                                                new_string, 
                                                set_preds, 
                                                ordering, 
                                                ids, 
                                                saved)
        
    new_string, ordering, saved = res[0],res[1], res[2]

    return new_string, ordering, saved

def add_non_leaf_negative_pred_to_digraph(pred, dnf, c, new_string, set_preds, \
                                                        ordering, ids, saved):
    """
    Update the digraph file with information on a non-leaf node for a negative
    predicate.
    """
    print('Negative')
    print(pred)
    print(ordering[pred])
    print('\n')
    if ordering[pred] == 0:
        ordering[pred] = (2, ids[dnf[c+1]])
    elif ordering[pred] == 1:
        ordering[pred] = 3
        params = '  [headlabel="No", labelangle=-45, labeldistance="2.5"];\n'
        new_string += ids[pred] + '-> ' + ids[dnf[c+1]] + params
    elif ordering[pred] == 3:
        pass
    else:
        num, tag = ordering[pred]
        assert num in [2,4], 'The ordering should be 2/4 but is {}'.format(num)
        if num == 2:
            pass
        if num == 4:
            params = '  [headlabel="Yes", labelangle=45, labeldistance="2.5"];\n'
            new_string += ids[pred] + '-> ' + ids[dnf[c+1]] + params
            params = '  [headlabel="No", labelangle=-45, labeldistance="2.5"];\n'
            new_string += ids[pred] + '-> ' + tag + params
    return new_string, ordering, saved

def add_leaf_negative_pred_to_digraph(pred, new_string, set_preds, \
                                            ordering, ids, saved):
    """
    Update the digraph file with information on a leaf node for a negative
    predicate.
    """
    saved += 1
    params = ' [fillcolor=forestgreen, fontcolor=green, label="CLICK IT"];\n'
    new_string += str(len(set_preds)+saved) + params
    if ordering[pred] == 0:
        ordering[pred] = (4, str(len(set_preds)+saved))
        
    elif ordering[pred] == 1:
        ordering[pred] = 3
        params = '  [headlabel="No", labelangle=-45, labeldistance="2.5"];\n'
        new_string += ids[pred] + '-> ' + str(len(set_preds)+saved) + params
        
    elif ordering[pred] == 3:
        pass
    
    else:
        num, tag = ordering[pred]
        assert num in [2,4], 'The ordering should be 2/4 but is {}'.format(num)
        if num == 2:
            ordering[pred] = 3
            params = '  [headlabel="No", labelangle=-45, labeldistance="2.5"];\n'
            new_string += ids[pred] + '-> ' + str(len(set_preds)+saved) + params
        if num == 4:
            pass
        
    return new_string, ordering, saved

def add_negative_pred_to_digraph(pred, dnf, c, num_preds, new_string, \
                                    set_preds, ordering, ids, saved):
    """
    Update the digraph file with information on a node for a negative predicate.
    """
    if c+1 < num_preds:
        res = add_non_leaf_negative_pred_to_digraph(pred, 
                                                    dnf, 
                                                    c, 
                                                    new_string, 
                                                    set_preds, 
                                                    ordering, 
                                                    ids, 
                                                    saved)
    else:
        res = add_leaf_negative_pred_to_digraph(pred, 
                                                new_string, 
                                                set_preds, 
                                                ordering, 
                                                ids, 
                                                saved)
        
    new_string, ordering, saved = res[0], res[1], res[2]
    
    return new_string, ordering, saved

def add_dnf_to_digraph(dnf, new_string, set_preds, ordering, ids, saved):
    """
    Update the digraph file with a set of nodes and relations imposed by a dnf.
    """
    num_preds = len(dnf)
    for c in range(num_preds):
        for pred in set_preds:     
            if pred == dnf[c]:
                res = add_positive_pred_to_digraph(pred, 
                                                   dnf, 
                                                   c, 
                                                   num_preds, 
                                                   new_string, 
                                                   set_preds, 
                                                   ordering, 
                                                   ids, 
                                                   saved)
                
            elif 'not(' + pred + ')' == dnf[c]:
                res = add_negative_pred_to_digraph(pred, 
                                                   dnf, 
                                                   c, 
                                                   num_preds, 
                                                   new_string, 
                                                   set_preds, 
                                                   ordering, 
                                                   ids, 
                                                   saved)
                
        new_string, ordering, saved = res[0], res[1], res[2]
                
    return new_string, ordering, saved

def add_singular_preds_to_digraph(set_preds, ordering, ids, new_string, saved):
    """
    Update the digraph file with information on nodes that stand for predicates
    which only appear in negative or positive form in a dnf.
    """
    for pred in set_preds:
        if ordering[pred] == 3:
            continue
        
        elif ordering[pred] == 1:
            ## a predicate only appears in the tree in its positive form
            saved += 1
            params = ' [fillcolor=tomato, fontcolor=black, label="DON\'T CLICK IT"];\n'
            new_string += str(len(set_preds)+saved) + params
            params = '  [headlabel="No", labelangle=-45, labeldistance="2.5"];\n'
            new_string += ids[pred] +'-> ' + str(len(set_preds)+saved) + params
            ordering[pred] = 3
            
        elif ordering[pred] == 0:
            raise Exception('Predicate {} not encountered in the '.format(pred) \
                            + 'program')
        
        else:
            ## a predicate only appears in the tree in its negated form
            two, tag = ordering[pred]
            assert two in [2,4], 'Checking the ordering: the ordering should \
                                  be 2/4 but is {}'.format(two)
            saved += 1
            params = ' [fillcolor=tomato, fontcolor=black, label="DON\'T CLICK IT"];\n'
            new_string += str(len(set_preds)+saved) + params
            params = '  [headlabel="Yes", labelangle=45, labeldistance="2.5"];\n'
            new_string += ids[pred] + ' -> ' + str(len(set_preds)+saved) + params
            params = '  [headlabel="No", labelangle=-45, labeldistance="2.5"];\n'
            new_string += ids[pred] + ' -> ' + tag + params
            ordering[pred] = 3
            
    return new_string, ordering, saved

def pretty_print(preds_in_dnfs):
    """
    Format the initial dnf to increase its readability.
    """
    pretty_program = ''
    for dnf in preds_in_dnfs:
        for pred in dnf:
            pretty_program += prettify(pred) + ' AND '
        pretty_program = pretty_program[:-5] + '\n'
        pretty_program += '          OR          \n'
    pretty_program = pretty_program[:-22]
    return pretty_program

def dnf2tree(program_string, expr=True):
    """
    Turn a string encoding a dnf into a digraph file encoding a decision tree.
    """
    preds_in_dnfs = dnf2conj_list(program_string)
    set_preds = get_used_preds(preds_in_dnfs)
    preds_in_dnfs, set_preds = check_for_additional_nodes(preds_in_dnfs, 
                                                          set_preds)
    new_string = 'digraph Tree {\ngraph [ordering="out"]\nnode [color="black", \
                  fontname=helvetica, shape=box, style="filled, rounded"];\
                  \nedge [fontname=helvetica];\n'
    res = set_digraph_ID_and_create_visited_nodes_dict(new_string, 
                                                       set_preds, 
                                                       expr)
    ids, ordering, new_string = res[0], res[1], res[2]

    saved = -1
    for dnf in preds_in_dnfs:
        new_string, ordering, saved = add_dnf_to_digraph(dnf, 
                                                         new_string, 
                                                         set_preds, 
                                                         ordering, 
                                                         ids, 
                                                         saved)
    new_string, ordering, saved = add_singular_preds_to_digraph(set_preds, 
                                                                ordering, 
                                                                ids, 
                                                                new_string, 
                                                                saved)    
    new_string += '}'

    print(new_string)
    print('\n')

    pretty_program = pretty_print(preds_in_dnfs)
    print(pretty_program)

    return new_string, pretty_program
        

if __name__ == "__main__":

    cwd = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d',
                        type=str,
                        help="Name of the file with the decision tree to visualize.",
                        default='TRIAL___')
    parser.add_argument('--only_dot', '-o',
                        type=bool,
                        help="Whether to only visualize a tree from the .dot " \
                            +"file or use data provided as the input to derive " \
                            +"the .dot file.",
                        default=False)

    args = parser.parse_args()

    folder_path = cwd+'/interprets_formula/'
    file_name = args.data

    if args.only_dot:
        print("Loading the new dot...")
        graph = pydotplus.graph_from_dot_file('new_tree.dot')

        print("Saving as an image...")
        graph.write_png(folder_path + "visualization/new_tree.png")
        sys.exit()

    beg = 0
    if re.search('\\/', file_name):
        name = re.search('\\/.+\\.', file_name).group(0)
        beg = 1
    else:
        name = file_name[:-3]

    print("Retrieving the program and the tree...")
    with open(folder_path + file_name, 'rb') as handle:
        program = pickle.load(handle)
    if 'program' in program.keys():
        program_string = program['program']
    else:
        program_string = list(program.keys())[0]

    ## expr=False generates a tree with raw predicates in the nodes
    new_string, pretty_program = dnf2tree(program_string, expr=False)

    with open(folder_path + "visualization/pretty_program_" + name[beg:-1] \
              + ".txt", "w") as text_file:
        text_file.write(pretty_program)

    print("Saving changes to dot...")
    with open('new_tree.dot', 'w') as file:
        file.write(new_string)

    print("Loading the new dot...")
    graph = pydotplus.graph_from_dot_file('new_tree.dot')

    print("Saving as an image...")
    graph.write_png(folder_path + "visualization/" + name[beg:-1] + ".png")
