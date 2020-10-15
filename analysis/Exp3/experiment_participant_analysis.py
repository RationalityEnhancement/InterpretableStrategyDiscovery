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

MAX_BONUS = 6
AGREEMENT_TYPE = 'clicks'
BONUS_RATE = 0.1
NUM_TESTING_TRIALS = 10
COND0 = 0
COND1 = 1

def get_goodbad_trial2(sequence, plp_tree):

    clicks = []

    #print([s[1] for s in sequence])
    for pair in sequence:
        st = should_terminate(plp_tree, pair[0])
        if not st and pair[1] == 0:
            init_state = pair[0]
            early_termination = True
        elif st and pair[1] == 0:
            clicks.append(1)
        elif plp_tree(pair[0], pair[1]):
            clicks.append(1)
        else:
            clicks.append(0)

    return clicks

def get_goodbad_participant2(participant_data, plp_tree):
    clicks_participant = []

    participant_state_actions = get_state_actions(participant_data)
    for sequence in participant_state_actions:

        clicks  = get_goodbad_trial2(sequence, plp_tree)
        clicks_participant.append(clicks)

    return clicks_participant


def get_goodbad_trial(sequence, plp_tree):

    good_clicks = 0
    bad_clicks = 0

    #print([s[1] for s in sequence])
    for pair in sequence:
        st = should_terminate(plp_tree, pair[0])
        if not st and pair[1] == 0:
            init_state = pair[0]
            early_termination = True
        elif st and pair[1] == 0:
            good_clicks += 1
        elif plp_tree(pair[0], pair[1]):
            good_clicks += 1
        else:
            bad_clicks += 1

    if (good_clicks + bad_clicks) == 0:
        bad_clicks += 1
    click_agreement_ratio = np.round(float(good_clicks) / (good_clicks + bad_clicks), 3)

    return click_agreement_ratio, good_clicks, bad_clicks

def get_goodbad_participant(participant_data, plp_tree):
    click_agreement_ratios = []
    good_clicks_part = 0
    bad_clicks_part = 0

    participant_state_actions = get_state_actions(participant_data)
    for sequence in participant_state_actions:

        click_agreement_ratio, good_clicks, bad_clicks  = get_goodbad_trial(sequence, plp_tree)
        click_agreement_ratios.append(click_agreement_ratio)
        good_clicks_part += good_clicks
        bad_clicks_part += bad_clicks

    return np.round(np.mean(click_agreement_ratios),3), good_clicks_part, bad_clicks_part




def get_agreement_trial(sequence, plp_tree):

    good_clicks = 0
    bad_clicks = 0
    early_termination = False
    mean_run_length = 0
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

    if early_termination:
        reset_value = [o.label for o in init_state.observed_nodes]
        tree_runs = [solve_mouselab(plp_tree, init_state, reset_value)[0] for _ in range(1000)]
        mean_run_length = np.mean([len(x) for x in tree_runs])
        bad_clicks += mean_run_length
    #print('good: {}, bad: {}'.format(good_clicks, bad_clicks))

    if bad_clicks == 0:
        sequencewise_agreement = 1
    else:
        sequencewise_agreement = 0
    click_agreement_ratio = np.round(float(good_clicks) / (good_clicks + bad_clicks), 3)

    return click_agreement_ratio, sequencewise_agreement, mean_run_length

def get_agreement_participant(participant_data, plp_tree):
    click_agreement_ratios = []
    sequencewise_agreements = []
    mean_run_lengths = []

    participant_state_actions = get_state_actions(participant_data)
    for sequence in participant_state_actions:

        click_agreement_ratio, sequencewise_agreement, mean_run_length  = get_agreement_trial(sequence, plp_tree)
        click_agreement_ratios.append(click_agreement_ratio)
        sequencewise_agreements.append(sequencewise_agreement)
        mean_run_lengths.append(mean_run_length)

    return click_agreement_ratios, mean_run_lengths

def get_agreement_sample(data, plp_tree):
    click_agreement_ratios_sample = []
    click_agreement_means_sample = []

    mean_run_lengths_sample = []

    for participant in data:
        click_agreement_ratios, mean_run_lengths  = get_agreement_participant(participant, plp_tree)

        click_agreement_ratios_sample.append(click_agreement_ratios)
        click_agreement_means_sample.append(np.round(np.mean(click_agreement_ratios),2))

        mean_run_lengths_sample.append(mean_run_lengths)


    return {'click_agreement_ratios_sample': click_agreement_ratios_sample,
            'click_agreement_means_sample': click_agreement_means_sample,
            'mean_run_lengths_sample': mean_run_lengths_sample}


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




def print_ttest(rews_group1, rews_group2):
    t_statistic, pval = stats.ttest_ind(rews_group1,rews_group2)
    print('t: {}, p_val: {}'.format(t_statistic, pval))

def print_whitney(rews_group1, rews_group2):
    u_statistic, pval = stats.mannwhitneyu(rews_group1,rews_group2)
    print('u: {}, p_val: {}'.format(u_statistic, pval))
    
def print_statistics(df1, df2, key):
    print(' -- df1 -- ')
    print('Median: {}'.format(r(np.median(df1[key]))))
    print('Mean: {}'.format(r(np.mean(df1[key]))))
    print('Std: {}'.format(r(np.std(df1[key]))))
    print('(${}\% \; (M= {}\%, SD = {}\%)$)'.format( r(np.median(df1[key])) , r(np.mean(df1[key])), r(np.std(df1[key]))))
    print()

    print(' -- df2 -- ')
    print('Median: {}'.format(r(np.median(df2[key]))))
    print('Mean: {}'.format(r(np.mean(df2[key]))))
    print('Std: {}'.format(r(np.std(df2[key]))))
    print('(${}\% \; (M= {}\%, SD = {}\%)$)'.format( r(np.median(df2[key])) , r(np.mean(df2[key])), r(np.std(df2[key]))))    
    print()

def r(number, digit=4):
    return round(number,digit)

def create_barplot(df1, df2, key, bins):
    
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
     