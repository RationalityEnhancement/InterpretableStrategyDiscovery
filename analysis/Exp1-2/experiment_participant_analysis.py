import json
import os
import argparse
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import copy
from hyperparams import STRATEGY
from PLP.DSL import *
from strategy_demonstrations import make_modified_env
from decision_tree_imitation_learning import solve_mouselab
from scipy import stats

MAX_BONUS = 1.5
AGREEMENT_TYPE = 'clicks'
BONUS_RATE = 0.02
NUM_TESTING_TRIALS = 10
COND0 = 0
COND1 = 1

def which_condition(num):
    if num == COND0:
        return 'EXPERIMENTAL'
    elif num == COND1:
        return 'CONTROL'

def json2dict(data):
    """
    Turn a json database file into a dicitonary
    """
    sub_all = data.get("values")
    keys = data.get("fields")
    dic_all = []
    for sub in sub_all:
        x = dict(zip(keys, sub))
        if not x['datastring'] is None:
            x['datastring'] = json.loads(x.get('datastring'))
        dic_all.append(x)
    return dic_all
    
def is_testing_data(cond_dic, subject):
    is_none = cond_dic[subject]['datastring'] is None
    if not is_none:
        is_debug = 'status' not in cond_dic[subject]['datastring'].keys()
    else:
        is_debug = False
    if is_none or is_debug:
        return False
    return True

def choose_condition(dic_all, cond):
    """
    Given condition number, extract participant data belonging
    to that condition so that it is testing data and the participant
    submitted the HIT/we accepted it.
    """
    all_condition_participants_data = []
    condition_participants_data = []
    for x in dic_all:
        if int(x['datastring']['questiondata']['condition']) == cond:
            all_condition_participants_data.append(x)
    print('LEN ' + which_condition(cond) + ' COND {}: {}'.format(cond, len(all_condition_participants_data)))

    def is_submit_or_accept(status):
        return status == 4 or status == 5

    for i in range(len(all_condition_participants_data)):
        _, _, status = participant_input_for_bonus_payment(cond_dic=data_dict, 
                                                           subject=i, 
                                                           condition=cond, 
                                                           out='info')
        print(status)
        if is_testing_data(all_condition_participants_data, i) and \
           is_submit_or_accept(status):
            condition_participants_data.append(all_condition_participants_data[i])
    return condition_participants_data

def get_participant_data(cond_dic, subject):
    """
    Output data important for the analyses and bonus payment out of
    a dictionary of data for participant in some condition.
    """

    if not is_testing_data(cond_dic, subject):
        return None, None, None, None, None
    worker_id = cond_dic[subject]['datastring']['workerId']
    hit_id = cond_dic[subject]['datastring']['assignmentId']
    worker_status = cond_dic[subject]['status']

    participant_data = []
    participant_answers = []
    data = cond_dic[subject]['datastring']['data']
    for x in data:
        trial_type = x['trialdata']['trial_type']
        try:
            block_type = x['trialdata']['block']
        except:
            block_type = ''
        if trial_type == 'mouselab-mdp' and block_type == 'testing':
            participant_data.append(x)
        elif trial_type == 'survey-text':
            participant_answers.append(x)

    return participant_data, participant_answers, hit_id, worker_id, worker_status

def see_participant_answers(cond_dic, subject):
    """
    Print subject answers to the most important questions of interpretability experiment.
    """
    _, answers, hit_id, worker_id, worker_status = get_participant_data(cond_dic, subject)
    print('{}:\n\n'.format(worker_id))
    if len(answers == 4):
        mapping[0] = 'Understood?: {}'
        mapping[1] = 'Explanation: {}'
    else:
        mapping[0] = 'Something unimportant: {}'
        mapping[1] = mapping[0]
    for i in range(len(answers)):
        print(mapping[i].format(a))
            
def participant_input_for_bonus_payment(cond_dic, subject, condition, plp_tree=None, 
                                        out='info'):
    """
    Output data needed to perform bonus payment to input subject with boto3 pacakge/interface.
    """
    assert out in ['info', 'bonus'], 'Only HIT/worker info or the bonus can' \
                                     + ' be outputted'
    p_data, _, hit_id, worker_id, status = get_participant_data(cond_dic, subject)
    if out == 'info':
        return hit_id, worker_id, status
    elif is_testing_phase(cond_dic, subject):
        bonus = calculate_bonus(p_data, plp_tree, condition)
    else:
        bonus = 0
    return bonus

def is_testing_phase(cond_dic, subject):
    p_data, _, _, _, _ = get_participant_data(cond_dic, subject)
    if p_data == [] or p_data is None:
        return False
    return True

def extract_reward(participant_data):
    """
    Output a list of rewards obtained in the test block for a given participant.
    """
    rewards = []
    for x in participant_data:
        rewards.append(x['trialdata']['score'])
    return np.array(rewards)

def get_state_actions(participant_data):
    """
    Convert participant data into a format acceptable by the PLP decision tree.
    """
    participant_state_actions = []
    for x in participant_data:
        list_of_strings = x['trialdata']['queries']['click']['state']['target']
        list_of_actions = [int(s) for s in list_of_strings] + [0]
        
        ground_truth_string = x['trialdata']['stateRewards']
        ground_truth_string = [0 if s == '' else s for s in ground_truth_string]
        ground_truth = [int(s) for s in ground_truth_string]
        initial_state = make_modified_env(ground_truth=ground_truth)
        
        sequence = list(STRATEGY(initial_state, list_of_actions))
        participant_state_actions.append(sequence)
    return participant_state_actions

def should_terminate(plp_tree, state):
    for i in range(12):
        if plp_tree(state, i+1):
            return False
    return True

def participant_agreement(participant_data, plp_tree, rew=False, just_rew=False):
    """
    For a given participant: 
    
      Calculate the percentage of clicks that agree with the PLP decision tree 
      and the percentage of whole click-sequences that exhibit that property.
    
      Optionally, output statistics connected to the gathered total reward.
    """
    pairwise_agreements = []
    sequencewise_agreement = 0
    participant_state_actions = get_state_actions(participant_data)
    rewards_list = extract_reward(participant_data)
    
    mean_reward = sum(rewards_list) / NUM_TESTING_TRIALS
    std_reward = np.std(rewards_list)
    var_reward = np.var(rewards_list)
    
    if just_rew:
        return mean_reward, std_reward, var_reward
    
    for sequence in participant_state_actions:
        
        good_clicks = 0
        bad_clicks = 0
        early_termination = False
        #print([s[1] for s in sequence])
        for pair in sequence:
            if not should_terminate(plp_tree, pair[0]) and pair[1] == 0:
                init_state = pair[0]
                early_termination = True
            elif should_terminate(plp_tree, pair[0]) and pair[1] == 0:
                good_clicks += 1
            elif plp_tree(pair[0], pair[1]):
                good_clicks += 1
            else:
                bad_clicks += 1
                
            ''' lambda st, act: is_on_highest_expected_value_path(st, act)
            for i in range(13):
                r = []
                for o in pair[0].node_map.values():
                    if o.observed:
                        r.append(o.value)
                    else: 
                        r.append(0)
                print('State {}: action {}: {}'.format(r, i, plp_tree(pair[0], i)))
            print('-------')
            '''
            
        if early_termination:
            reset_value = [o.label for o in init_state.observed_nodes]
            tree_runs = [solve_mouselab(plp_tree, init_state, reset_value)[0] for _ in range(1000)]
            mean_run_length = np.mean([len(x) for x in tree_runs])
            bad_clicks += mean_run_length
        #print('good: {}, bad: {}'.format(good_clicks, bad_clicks))
        if bad_clicks == 0:
            sequencewise_agreement += 1
        click_agreement_ratio = float(good_clicks) / (good_clicks + bad_clicks)
        pairwise_agreements.append(click_agreement_ratio)

    proportion1 = float(sequencewise_agreement) / len(participant_state_actions)
    proportion2 = np.mean(pairwise_agreements)

    if rew:
        return proportion1, proportion2, mean_reward, std_reward, var_reward
    return proportion1, proportion2

def calculate_bonus(participant_data, plp_tree, condition, agreement=AGREEMENT_TYPE):
    """
    For a given participant: 
    
      Based on the clicks the participant has made or the reward he/she has got
      (dependent on the condition), calculate the bonus he/she should obtain.
      
    Bonus for condition 0 is the proportion of 'correct' clicks multipled by a
    maximum of the bonus payment.
    
    Bonus for condition 1 is the score, multiplied by a magical constant to get
    $8/h, roughly, and rounded to the nearest cent.
    """
    if condition == COND0:
        agreed_seqs, agreed_clicks = participant_agreement(participant_data, 
                                                           plp_tree)
        weighting = agreed_clicks if agreement == 'clicks' else agreed_seqs
    elif condition == COND1:
        mean_rew, _, _ = participant_agreement(participant_data, 
                                               plp_tree, 
                                               just_rew=True)
        not_rounded_weighting = mean_rew * BONUS_RATE
        weighting = np.round(not_rounded_weighting, 2)
        weighting = 0 if weighting < 0 else weighting
    return weighting * MAX_BONUS

def extract_high_bonus_participants(worker_ids, bonuses, threshold):
    """
    Return IDs of worker who got a bonus of at least 'threshold'.
    """
    zipped = zip(worker_ids, bonuses)
    ids = [id_ for id_, bonus in zipped if bonus >= threshold]
    return ids

def extract_high_agreement_participants(worker_ids, agreements, threshold):
    """
    Return IDs of worker who got a agreement score of at least 'threshold'.
    """
    zipped = zip(worker_ids, agreements)
    ids = [id_ for id_, agreement in zipped if agreement >= threshold]
    return ids

def calculate_agreement_statistics(all_participants_data, plp_tree, rew=False):
    """
    For participants in a given condition:
    
      Calculate percentages of clicks and sequences correct under the decision
      tree in each of the testing trials. Calculate their means.
    
      Optionally, output statistics connected to the gathered total reward.
    """
    def is_submit_or_accept(status):
        return status == 4 or status == 5
    
    all_rewards = []
    all_mean_rewards = []
    all_std_rewards = []
    all_seq_proportions = []
    all_pair_proportions = []
    num_data = len(all_participants_data)
    for i in range(len(all_participants_data)):
        if is_testing_phase(all_participants_data, i):
            participant_data, _, _, _, s = get_participant_data(all_participants_data, i)
            if is_submit_or_accept(s):
                rewards_list = extract_reward(participant_data)
                mean_reward = np.mean(rewards_list)
                sum_reward = sum(rewards_list)
                std_reward = np.std(rewards_list)
                seq_proportion, pair_proportion = participant_agreement(participant_data,
                                                                        plp_tree)
                all_mean_rewards.append(mean_reward)
                all_rewards.append(sum_reward)
                all_std_rewards.append(std_reward)
                all_seq_proportions.append(seq_proportion)
                all_pair_proportions.append(pair_proportion)
    mean_reward = np.mean(all_rewards)
    mean_std_reward = np.mean(all_std_rewards)
    mean_seq_proportion = np.mean(all_seq_proportions)
    mean_pair_proportion = np.mean(all_pair_proportions)
    std_seq_proportion = np.std(all_seq_proportions)
    std_pair_proportion = np.std(all_pair_proportions)

    reward_stats = (all_rewards, all_mean_rewards)
    cond0_stats = [all_seq_proportions, all_pair_proportions]
    cond0_mean_seq_stats = [mean_seq_proportion, std_seq_proportion]
    cond0_mean_pair_stats = [mean_pair_proportion, std_pair_proportion]
    cond0_mean_stats = [cond0_mean_seq_stats, cond0_mean_pair_stats]

    if rew:
        return reward_stats, cond0_stats, cond0_mean_stats
    return cond0_stats, cond0_mean_stats

#def calculate_reward_statistics(all_participants_data):
#    """
#    Output statistics connected to the gathered total reward.
#    """
#    all_rewards = []
#    all_std_rewards = []
#    all_var_rewards = []
#    num_data = len(all_participants_data)
#    for participant_data in all_participants_data:
#        rewards_list = extract_reward(participant_data)
#        mean_reward = np.mean(rewards_list)
#        std_reward = np.std(rewards_list)
#        var_reward = np.var(rewards_list)
#
#        all_rewards.append(mean_reward)
#        all_std_rewards.append(std_reward)
#        all_var_rewards.append(var_reward)
#    mean_reward = np.mean(all_rewards)
#    mean_var_reward = np.mean(all_var_rewards)
#    mean_std_reward = np.mean(all_std_rewards)
#    reward_stats = [mean_reward, mean_std_reward, mean_var_reward]
#    return reward_stats

def generate_input_for_bonus_payment(data_dict, condition, plp_tree=None, hit_id=None):
    """
    Output data needed to perform bonus payment to all the subjects from the HIT.
    """
    hitIDs = []
    workerIDs = []
    statuses = []
    bonuses = []
    
    if condition == COND0:
        assert plp_tree is not None, 'Need to specify plp_tree for condition 0'

    def is_correct_hitID(got_hitID, wanted_hitID):
        if wanted_hitID == None:
            return got_hitID != None
        return got_hitID == wanted_hitID

    def is_submit_or_accept(status):
        return status == 4 or status == 5

    for i in range(len(data_dict)):
        res = participant_input_for_bonus_payment(cond_dic=data_dict, 
                                                  subject=i, 
                                                  condition=condition, 
                                                  out='info')
        worker_hitid, worker_id, worker_status = res[0], res[1], res[2]
        if is_correct_hitID(worker_hitid, hit_id) and is_submit_or_accept(worker_status):
            hitIDs.append(worker_hitid)
            workerIDs.append(worker_id)
            worker_status = 5 if worker_status == 4 else worker_status
            statuses.append(worker_status)
            bonus = participant_input_for_bonus_payment(cond_dic=data_dict, 
                                                        subject=i, 
                                                        condition=condition, 
                                                        plp_tree=plp_tree, 
                                                        out='bonus')
            bonus = 0 if bonus < 0 else bonus

            bonuses.append(bonus)
            print("Assignment {}. {}: status: {}, bonus: {}".format(worker_hitid, 
                                                             worker_id, 
                                                             worker_status, 
                                                             bonus))
    return hitIDs, workerIDs, statuses, bonuses

def generate_statistics(cond_dic, condition, plp_tree, hit_id=None):
    """
    Produce the lists of rewards needed for statistical comparisons of effect
    and (for condition 0) agreements statistics to quantify people's understanding.
    """
#    if condition == COND0:
    assert plp_tree is not None, 'Need to specify plp_tree'
    res = calculate_agreement_statistics(cond_dic, plp_tree, rew=True)
    rews, agreements, mean_agreements = res[0], res[1], res[2]
    return rews, agreements, mean_agreements
#    elif condition == COND1:
#        rews = calculate_reward_statistics(cond_dic)
#        return rews

def plot_barplots(rews_group, title, add_info = '', cbrt=False):
    if 'EXPERIMENTAL' in title:
        if 'ewards' in title:
            filename = 'exp_rewards'
        elif 'greements' in title:
            filename = 'exp_agreements'
    elif 'CONTROL' in title:
        if 'ewards' in title:
            filename = 'cnt_rewards'
        elif 'greements' in title:
            filename = 'cnt_agreements'

    if not cbrt:
        filename += add_info + '.png'
        sns.distplot(rews_group)
    else:
        filename += '_cbrt' + add_info + '.png'
        sns.distplot(np.cbrt(rews_group))
    plt.title(title)
    plt.savefig(filename)
    plt.show()
    
def print_ttest(rews_group1, rews_group2):
    t_statistic, pval = stats.ttest_ind(rews_group1,rews_group2)
    print('t: {}, p_val: {}'.format(t_statistic, pval))

def print_whitney(rews_group1, rews_group2):
    u_statistic, pval = stats.mannwhitneyu(rews_group1,rews_group2)
    print('u: {}, p_val: {}'.format(u_statistic, pval))
    
def print_statistics(dics, plp_tree, add_info=''):
    condition_tree_dict, condition_notree_dict = dics
    rews, agreements, mean_agreements = generate_statistics(condition_tree_dict, 
                                                            COND0, 
                                                            plp_tree)
    total_rews_list, mean_rews_list = rews
    seq_agreements, pair_agreements = agreements
    mean_seq_agreements, mean_pair_agreements = mean_agreements
    print('\n\n')
    rews, agreements, mean_agreements = generate_statistics(condition_notree_dict, 
                                                            COND1, 
                                                            plp_tree)
    total_rews_list2, mean_rews_list2 = rews
    seq_agreements2, pair_agreements2 = agreements
    mean_seq_agreements2, mean_pair_agreements2 = mean_agreements
    print('\n\n')
    print('                  EXPERIMENTAL CONDITION:\n\n')
    print('Num participants: {}'.format(len(pair_agreements)))
    print('Click-agreements per participant: {}'.format(pair_agreements))
    print('Mean, std of click-agreements: {}\n'.format(mean_pair_agreements))
    print('Seq-agreements per participant: {}'.format(seq_agreements))
    print('Mean, std of seq-agreements: {}\n'.format(mean_seq_agreements))
    print('List of collected total rewards: {}'.format(total_rews_list))
    print('List of collected mean rewards: {}'.format(mean_rews_list))
    print('\nMean, std total reward per condition: {}'.format([np.mean(total_rews_list), 
                                                               np.std(total_rews_list)]))
    print('                  CONTROL CONDITION:\n\n')
    print('Num participants: {}'.format(len(pair_agreements2)))
    print('Click-agreements per participant: {}'.format(pair_agreements2))
    print('Mean, std of click-agreements: {}\n'.format(mean_pair_agreements2))
    print('Seq-agreements per participant: {}'.format(seq_agreements2))
    print('Mean, std of seq-agreements: {}\n'.format(mean_seq_agreements2))
    print('List of collected total rewards: {}'.format(total_rews_list2))
    print('List of collected mean rewards: {}'.format(mean_rews_list2))
    print('\nMean, std total reward per condition: {}'.format([np.mean(total_rews_list2), 
                                                               np.std(total_rews_list2)]))
    
    plot_barplots(total_rews_list, 'Rewards histogram for the EXPERIMENTAL group', add_info)
    plot_barplots(total_rews_list,  'Cube root rewards histogram for the EXPERIMENTAL group', add_info, cbrt=True)
    plot_barplots(total_rews_list2, 'Rewards histogram for the CONTROL group', add_info)
    plot_barplots(total_rews_list2, 'Cube root rewards histogram for the CONTROL group', add_info, cbrt=True)
    
    print('\nt-test for total rewards:')
    print_ttest(total_rews_list, total_rews_list2)
    print('Mann-Whitney u-test for total rewards:')
    print_whitney(total_rews_list, total_rews_list2)

    plot_barplots(pair_agreements, 'Agreements histogram for the EXPERIMENTAL group', add_info)
    plot_barplots(pair_agreements, 'Cube root agreements histogram for the EXPERIMENTAL group', add_info, cbrt=True)
    plot_barplots(pair_agreements2, 'Agreements histogram for the CONTROL group', add_info)
    plot_barplots(pair_agreements2, 'Cube root agreements histogram for the CONTROL group', add_info, cbrt=True)

    print('\nt-test for click-agreements:')
    print_ttest(pair_agreements, pair_agreements2)
    print('Mann-Whitney u-test for click-agreements:')
    print_whitney(pair_agreements, pair_agreements2)
    
    return {'pair_agreement': pair_agreements+pair_agreements2 ,
             'seq_agreement': seq_agreements+seq_agreements2,
            'total_rew': total_rews_list+total_rews_list2,
            'mean_rew':mean_rews_list+mean_rews_list2}
    
def print_input_for_bonus_payment(dics, plp_tree):
    hitIDs, workerIDs, statuses, bonuses = generate_input_for_bonus_payment(dics[0], COND0, plp_tree)
    hitIDs2, workerIDs2, statuses2, bonuses2 = generate_input_for_bonus_payment(dics[1], COND1)
    text = 'assignmentIDs = {}\n'.format(hitIDs+hitIDs2) \
         + 'workerIDs = {}\n'.format(workerIDs+workerIDs2) \
         + 'statuses = {}\n'.format(statuses+statuses2) \
         + 'bonuses = {}\n'.format(bonuses+bonuses2)
    print(text)
    
    return { 'assignmentIDs': hitIDs+hitIDs2,
            'workerIDs': workerIDs+workerIDs2,
             'bonuses': bonuses+bonuses2
           }

def _cond0_answers(df):
    print('            ANSWERS:\n')
    ans_list = df[df['condition']==0]['qDes'].tolist()
    for a in ans_list:
        print(a)
    print('\n')



def test_participant_behavior(participant_data, plp_tree):
    """

    """

    participant_state_actions = get_state_actions(participant_data)
    rewards_list = extract_reward(participant_data)
    pairwise_agreements = []
    
    # loop through ten trials
    for sequence in participant_state_actions:
        
        good_clicks = 0
        bad_clicks = 0
        early_termination = False
        #print([s[1] for s in sequence])
        #print(sequence)
        
        for pair in sequence:
            #print(pair[0]) # -> one state
            #print(pair[1])
            #print(plp_tree(pair[0], pair[1]) == 1)
            #print('')
            
            if plp_tree(pair[0], pair[1]):
                good_clicks += 1
            else:
                bad_clicks += 1
                
            # lambda st, act: is_on_highest_expected_value_path(st, act)
            for i in range(13):
                r = []
                for o in pair[0].node_map.values():
                    if o.observed:
                        r.append(o.value)
                    else: 
                        r.append(0)
                #print('State {}: action {}: {}'.format(r, i, plp_tree(pair[0], i)))
            #print('-------')
            
            
        #print('good: {}, bad: {}'.format(good_clicks, bad_clicks))
        #print('')
        #print('')
        #print('')

        click_agreement_ratio = float(good_clicks) / (good_clicks + bad_clicks)
        pairwise_agreements.append(click_agreement_ratio)


    proportion2 = np.mean(pairwise_agreements)

    return proportion2





     
def print_statistics2(df1, df2, key):
    print(' -- df1 -- ')
    print('Mean: {}'.format(np.mean(df1[key])))
    print('Std: {}'.format(np.std(df1[key])))
    print()
    
    print(' -- df2 -- ')
    print('Mean: {}'.format(np.mean(df2[key])))
    print('Std: {}'.format(np.std(df2[key])))
    print() 
    
    print(' -- test-- ')
    print_ttest(df1[key], df2[key])
    print_whitney(df1[key], df2[key])
    print()

def create_barplot2(df1, df2, key, bins):
    
    exp = df1[key]
    cont = df2[key]
    exp_norm = plt.hist([exp, cont], bins, histtype='bar')[0][0]
    cont_norm = plt.hist([exp, cont], bins, histtype='bar')[0][1]
    exp_norm = exp_norm /sum(exp_norm)
    cont_norm = cont_norm /sum(cont_norm)
    
    exp_err_up = []
    exp_err_down = []
    for x in exp_norm:
        sd = np.sqrt(x*(1-x))
        err = sd/np.sqrt(len(df1))
        err = 1.96*err
        exp_err_up.append(err)
        if err > x:
            err = x
        exp_err_down.append(err)          

    cont_err_up = []
    cont_err_down = []
    for x in cont_norm:
        sd = np.sqrt(x*(1-x))
        err = sd/np.sqrt(len(df2))
        err = 1.96*err
        cont_err_up.append(err)
        if err > x:
            err = x
        cont_err_down.append(err)   
       
    dif = abs(bins[0]-bins[1])/2
    pos = []
    for x in bins[0:len(bins)-1]:
        pos.append(x + dif)
    
    print(pos)
    plt.clf() 
    plt.bar( pos, exp_norm, yerr=[exp_err_down, exp_err_up],  width= dif*0.7, align='edge', label="Flowchart", capsize=4, error_kw=dict(lw=1))
    plt.bar( pos, cont_norm, yerr=[cont_err_down, cont_err_up], width= dif*-0.7, align='edge', label="No Flowchart", capsize=4, error_kw=dict(lw=1))

    plt.xlim((min(bins), max(bins)))
    plt.xticks(bins)
    plt.legend()
    
    
    



def test_participant_behavior2(participant_data, plp_tree):
    """

    """

    participant_state_actions = get_state_actions(participant_data)
    rewards_list = extract_reward(participant_data)
    
    term = 0
    count = 0
    best_path_clicks = 0
    bad_clicks = 0
    
    leaf_clicks = 0
    
    # loop through ten trials
    for sequence in participant_state_actions:
        
        good_clicks = 0
        ten_indeces = []
        #print([s[1] for s in sequence])
        #print(sequence)
        
        for pair in sequence:
                
            for i in range(13):
                r = []
                for o in pair[0].node_map.values():
                    if o.observed:
                        r.append(o.value)
                    else: 
                        r.append(0)
            
            indices = [i for i, x in enumerate(r) if x == 10]
            
            if len(indices) > 0 and len(indices) > len(ten_indeces):
                index = indices[len(indices)-1]
                
                if index in [1,2]:
                    count += 1
                    if pair[1] in [3,4]:
                        leaf_clicks += 1
                    elif pair[1] in [1,2]:
                        best_path_clicks += 1
                    elif pair[1] == 0:
                        term += 1
                    else:
                        bad_clicks += 1

                elif index in [5,6]:
                    count += 1
                    if pair[1] in [7,8]:
                        leaf_clicks += 1
                    elif pair[1] in [5,6]:
                        best_path_clicks += 1
                    elif pair[1] == 0:
                        term += 1
                    else:
                        bad_clicks += 1
               
                elif index in [9,10]:
                    count += 1
                    if pair[1] in [11,12]:
                        leaf_clicks += 1
                    elif pair[1] in [5,6]:
                        best_path_clicks += 1
                    elif pair[1] == 0:
                        term += 1
                    else:
                        bad_clicks += 1
                        
                    #if plp_tree(pair[0], pair[1]):
 
                        
            ten_indeces = indices
        
            #print(pair[0]) # -> one state
            #print(pair[1])
            #print(plp_tree(pair[0], pair[1]) == 1)
            #print('')
            #print('State {}: action {}: {}'.format(r, i, plp_tree(pair[0], i)))
            #print('-------')
            
            
        #print('good: {}, bad: {}'.format(good_clicks, bad_clicks))
        #print('')
        #print('')
        #print('')
        

    return  {'term': term, 'leaf_clicks':leaf_clicks, 'best_path_clicks': best_path_clicks, 'bad_clicks': bad_clicks, 'count':count }
    
    
    
if __name__ == '__main__':
    
    cwd = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument('--tree_path', '-p', 
                        type=str,
                        help="Path to the file generated by the interpretation algorithm.")
    parser.add_argument('--experiment_data_path', '-d',
                        type=str,
                        help="Path to the json file generated by heroku.")

    args = parser.parse_args()
    print(args)
    
    with open(args.experiment_data_path, 'r') as handle:
        data = json.load(handle)

    with open(args.tree_path, 'rb') as handle:
        interpretation_dict = pickle.load(handle)

    
    print(interpretation_dict['program'])
    plp_tree = eval('lambda st, act : ' + interpretation_dict['program'])
    data_dict = json2dict(data)
    dataframe = make_dataframe(data_dict)
    
    tt = [int(a) for a in dataframe['totalTime'].to_list() if a != '-']
    print(tt)
    print(np.mean(tt) / 60)
    
    #ct = dataframe['cheatTrials'].to_list()
    #at = dataframe['attempts'].to_list()
    #plt.hist(at)
    #plt.title('Histogram of attempts needed to pass the quiz')
    #plt.show()
    #plt.hist(ct)
    #plt.title('Histogram of testing trials without any clicks')
    #plt.show()
    #import sys
    #sys.exit()

    for wid in ['AEQ6KLNVTRKKR', 'APXF9SKBQ90QS', 'ADORV9ANYTPAC', 'A35KXHDBZLE6XU', 'A2YK3N7R4XIETV', 'AK884VX5S74ZW', 'A37FVE8D81SVMJ', 'A2DDXYTEVDV5F5', 'A1TEXCUTI4IUUN']:
        early_part = df2list_row(dataframe[dataframe['WorkerId'] == wid].reset_index(drop=True))
        print(early_part)
        early_part_data, _, _, _, _ = get_participant_data(early_part, 0)
        bon = calculate_bonus(early_part_data, plp_tree, 1)
        print('{} (early quit): {}'.format(wid, bon))
    
    condition_tree_dict = dataframe[(dataframe.condition == COND0) & (dataframe.cheatTrials <= 9)]# & (dataframe.attempts <= 1)]
    print_cond0_answers(condition_tree_dict)
    condition_tree_dict = df2list_row(condition_tree_dict.reset_index(drop=True))

    condition_notree_dict = dataframe[(dataframe.condition == COND1) & (dataframe.cheatTrials <= 9)]# & (dataframe.attempts <= 1)]
    condition_notree_dict = df2list_row(condition_notree_dict.reset_index(drop=True))

    print_input_for_bonus_payment((condition_tree_dict, condition_notree_dict), plp_tree)
    
    print_statistics((condition_tree_dict, condition_notree_dict), plp_tree, add_info = '_cheat9')
    