from RL2DT.hyperparams import *
from RL2DT.strategy_demonstrations import reward
from RL2DT.formula_visualization import prettify
from hyperparams import *
from translation import alternative_procedures2text as translate
from formula_procedure_transformation import DNF2LTL, trajectorize, sizeBig, ConversionError
from formula_pruning import simplify_and_optimize
import argparse
import pickle
import os
import numpy as np
import sys
from RL2DT.demonstration_sampling import get_state_action_representation_from_DSL

        
def create_dir(file_path):
    """
        Create directory if it does not exist
    """
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def load_interpreted_data(formula_filename, preds_filename, demos_filename, path=None):
    """
    Load an input DNF formula along with all the predicates and the demos used 
    in creating the formula. 

    Parameters
    ----------
    formula_filename : str
        Name of the file with the formula to recover
    preds_filename : int 
        Name of the file with the predicate matrix
    demos_filename : int
        Name of the file with the demonstrations
        
    Returns
    -------
    formula : str
        DNF formula describing a planning strategy represented by demos
    pred_matrix : tuple
        A tuple with the data used in the AI-Interpret algorithm that finds the
        DNF formula
        data_matrix : np.array(X)
            Matrix with "active" predicates for every positive/negative 
            demonstration
        X : csr_matrix
            X.shape = (num_trajs, num_preds)
        y : [ int(bool) ]
            y.shape = (num_trajs,)
            Vector indicating whether the demonstration was positive or negative
    demos : dict
        pos: [ (IHP.modified_mouselab.Trial, int) ]
            Demonstrated (state, action) pairs
        neg: [ (IHP.modified_mouselab.Trial, int) ]
            (state, action) pairs for encountered states and non-optimal actions 
            according to the RL policy
        leng_pos: [ int ]
            Lengths of the consecutive demonstrations
        leng_neg: [ int ]
            Lengths of the negative examples for the consecutive demonstrations
    """
    cwd = os.getcwd()
    
    if not(path):
        formula_path = cwd+'/interprets/'
        folder_path = cwd+'/PLP_data/'
    else:
        folder_path = path
        formula_path = path
    
    print("Retrieving the formula...")
    with open(formula_path + formula_filename, 'rb') as handle:
        dict_object = pickle.load(handle)
    formula = dict_object
    print('Done')
    
    print("Retrieving the predicate matrix...")
    with open(folder_path + preds_filename, 'rb') as handle:
        pred_matrix = pickle.load(handle)
    print('Done')
        
    print("Retrieving the demos...")
    with open(folder_path + demos_filename, 'rb') as handle:
        demos = pickle.load(handle)
    print('Done')
    
    return formula, pred_matrix, demos
    
def extract_states_actions(trajs):
    """
    Turn a list of trajectories into an appropriate format with ground truth
    environments as one list and sequence of actions as another list.
    """
    states = [traj[0][0].ground_truth for traj in trajs]
    actions = [[traj[i][1] for i in range(len(traj))] for traj in trajs]
    return states, actions
    
def DNF2LTL_pruning(formula_name, preds_name, demos_name, path=None):
    """
    Wrapper function to produce a procedural description out of a DNF formula.
    
    It tries to find a description by first removing (all the) predicates which 
    were defined as redundant.
    
    For example, not(is_observed) and not(is_previous_observed_max) would not 
    have a procedural description different than itself but after removing the 
    second predicate, a procedural formula could be not(is_observed) UNTIL 
    is_previous_observed_max. 
    
    Reduncant predicates are selected by hand are are hyperparameters.
    
    If this fails, the function extracts a procedural description from the initial
    formula.
        
    Returns
    -------
    LTL_formula : str
        Procedural description in readable form
    raw_LTL_formula : str
        Procedural description in callable form (with lambdas, colons, etc.)
    c : int
        Complexity (the number of used predicates) in the description
    """
    formula, pred_matrix, demos = load_interpreted_data(formula_name, preds_name, demos_name, path)

    pretty_form = prettify(formula) if formula != None else formula
    print("\n\n\n\n\n" +\
          "COMPUTING PROCEDURE OUT OF {}: \n\n\n\n\n".format(pretty_form))
    trajs = trajectorize(demos)
    num_pos_demos = sum(pred_matrix[-1])
    pred_pos_matrix = pred_matrix[0][:num_pos_demos]
    
    envs, action_seqs = extract_states_actions(trajs)
    pipeline = [(BRANCHING, reward)] * len(envs)
    
    try:
        LTL_formula, c, raw_LTL_formula = DNF2LTL(
                                            phi=formula, 
                                            trajs=trajs, 
                                            predicate_matrix=pred_pos_matrix,
                                            allowed_predicates=ALLOWED_PREDS,
                                            redundant_predicates=ALLOWED_PREDS,
                                            redundant_predicate_types=REDUNDANT_TYPES,
                                            p_envs=envs,
                                            p_actions=action_seqs,
                                            p_pipeline=pipeline)
    except ConversionError:
        LTL_formula, c, raw_LTL_formula = DNF2LTL(phi=formula, 
                                                  trajs=trajs, 
                                                  predicate_matrix=pred_pos_matrix,
                                                  allowed_predicates=ALLOWED_PREDS,
                                                  redundant_predicates=[],
                                                  p_envs=envs,
                                                  p_actions=action_seqs,
                                                  p_pipeline=pipeline)
    print('\n\nPROCEDURAL FORMULA:\n\n{}\n\nComplexity: {}'.format(LTL_formula, c))
    
    proc_parts = LTL_formula.split('\n\nOR\n\n')
    raw_proc_parts = raw_LTL_formula.split('\n\nOR\n\n')
    
    new_proc_formula, raw_new_proc_formula = '', ''
    best_lmarglik = -np.inf
    for proc_formula, raw_proc_formula in zip(proc_parts, raw_proc_parts):
        res = simplify_and_optimize(formula=proc_formula, formula_raw=raw_proc_formula,
                                    envs=envs,
                                    action_seqs=action_seqs,
                                    pipeline=pipeline)
        part_proc_formula, part_raw_proc_formula, lmarglik = res
        if lmarglik > best_lmarglik:
            new_proc_formula = part_proc_formula
            raw_new_proc_formula = part_raw_proc_formula
            best_lmarglik = lmarglik
    c = sizeBig(new_proc_formula)
    proc_formula = new_proc_formula
    raw_proc_formula = raw_new_proc_formula
    
    print('\n\nOPTIIMIZED PROCEDURAL FORMULA:\n\n{}'.format(proc_formula) + \
          '\n\nComplexity: {}'.format(c))
    strategy_log = proc_formula
    ##strategy_log += '\n\n' + '-'*30 + '\n\n' + strategy_log
 
    text = translate(proc_formula)
    print('\n\nTranslation: {}\n'.format(text))
    strategy_log += '\n\nTranslation: {}\n'.format(text)
    
    with open('./farsighted_strategy/far_sighted_strategy', "w") as strategy:
        strategy.write(strategy_log)
    return strategy_log
    
if __name__ == "__main__":
    
    cwd = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument('--formula_name', '-f', 
                        help="Name of the interpretation algorithm to use.", 
                        default='strategy')
    parser.add_argument('--preds_name', '-r',
                        help="Whether to cluster DSL data hierarhically or " \
                            +"use a neural network and learn a hidden space.",
                        default='DSL')
    parser.add_argument('--demos_name', '-d',
                        help="Whether to find the candidates for the number of " \
                            +"clusters automatically or use the " \
                            +"candidate_clusters parameter.",
                        default='demos')
    parser.add_argument('--path', '-p',
                        help="Whether to find the candidates for the number of " \
                            +"clusters automatically or use the " \
                            +"candidate_clusters parameter.",
                        default='./farsighted_strategy')
   
    args = parser.parse_args()
    
    DNF2LTL_pruning(formula_name=args.formula_name,
                    preds_name=args.preds_name,
                    demos_name=args.demos_name,
                    path=args.path)

