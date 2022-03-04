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
    #print('')
    #print(' Neweeew Trial')

    good_clicks = 0
    bad_clicks = 0
    early_termination = False
    mean_run_length = 0

    plp_tree = eval(plp_tree)

    for pair in sequence:

        #print('')
        #print([node.label for node in pair[0].observed_nodes])
        #print([node.value for node in pair[0].observed_nodes])
        #print(pair[1])

        if not should_terminate(plp_tree, pair[0]) and pair[1] == 0:
            init_state = pair[0]
            early_termination = True
            #print('ET')
        elif should_terminate(plp_tree, pair[0]) and pair[1] == 0:
            good_clicks += 1
            #print('Good Term')
        elif plp_tree(pair[0], pair[1]):
            #print('good click')
            good_clicks += 1
        else:
            #print('bad click')
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
    counter = 0
    print(len(data))
    for participant in data:
        print(counter)
        counter += 1

        click_agreement_ratios, mean_run_lengths  = get_agreement_participant(participant, plp_tree)

        click_agreement_ratios_sample.append(click_agreement_ratios)
        click_agreement_means_sample.append(np.round(np.mean(click_agreement_ratios),4))

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
        list_of_strings = x['queries']['click']['state']['target']
        list_of_actions = [int(s) for s in list_of_strings] + [0]

        ground_truth_string = x['stateRewards']
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



# ------------------------- STATS ---------------------------------------------------------
def print_ttest(rews_group1, rews_group2):
    t_statistic, pval = stats.ttest_ind(rews_group1,rews_group2)
    print('t: {}, p_val: {}'.format(r(t_statistic), r(pval,digit=8)))

def print_whitney(rews_group1, rews_group2):
    u_statistic, pval = stats.mannwhitneyu(rews_group1,rews_group2)
    print('u: {}, p_val: {}'.format(r(u_statistic), r(pval,digit=8)))


def getPTable(dfs, key):
    cols = ['cond']
    for df in dfs:
        cols.append(str(df.condition_type.iloc[0]))

    df_p = pd.DataFrame(columns = cols)
    for df1 in dfs:
        res = []
        for df2 in dfs:
            res.append(round(stats.ttest_ind(df1[key], df2[key])[1], 3))
        df_p.loc[len(df_p)] = [str(df1.condition_type.iloc[0])] + res

    return df_p

def getUTable(dfs, key):
    cols = ['cond']
    for df in dfs:
        cols.append(str(df.condition_type.iloc[0]))

    df_p = pd.DataFrame(columns = cols)
    for df1 in dfs:
        res = []
        for df2 in dfs:
            res.append(round(stats.mannwhitneyu(df1[key], df2[key])[1], 3))
        df_p.loc[len(df_p)] = [str(df1.condition_type.iloc[0])] + res

    return df_p


def print_statistics(dfs, key, title):
    print(title)
    print('')

    for df in dfs:
        print(' -- ' + str(df.condition_type.iloc[0]) + ' -- ')
        print('Mean: {}'.format(r(np.mean(df[key]))))
        print('Std: {}'.format(r(np.std(df[key]))))
        print()

    print('')
    print_ttest(dfs[0], dfs[1])

    print('')
    print_whitney(dfs[0], dfs[1])

    print('')
    print_simple_barplots(dfs, key)

def print_simple_barplots(dfs, key):
    labels = []
    means = []
    errors = []
    for df in dfs:
        labels.append(df.conditionType.iloc[0])
        means.append(df[key].mean())
        errors.append(1.96*df[key].std()/np.sqrt(len(df)))

    plt.bar(x=range(len(dfs)), height=means, yerr=errors, tick_label= labels)


def r(number, digit=4):
    return round(number,digit)

def print_statistics2(df1, df2, key):
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

    print(' -- test-- ')
    print_ttest(df1[key], df2[key])
    print_whitney(df1[key], df2[key])
    print()

# ------------------------- PLOTS ---------------------------------------------------------
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
    plt.bar( pos, cont_norm, yerr=[cont_err_down, cont_err_up], width= dif*-0.7, align='edge', label="Action Feedback", capsize=4, error_kw=dict(lw=1))

    plt.xlim((min(bins), max(bins)))
    plt.xticks(bins)
    plt.legend()



def create_barplot2(df, cond1, cond2, key, bin_number):

    exp  = (df[df.condition_type == cond1][key]/10).tolist()
    cont = (df[df.condition_type == cond2][key]/10).tolist()
    bins = getBins(exp + cont, bin_number)

    exp_norm = plt.hist([exp, cont], bins, histtype='bar')[0][0]
    cont_norm = plt.hist([exp, cont], bins, histtype='bar')[0][1]
    exp_norm = exp_norm /sum(exp_norm)
    cont_norm = cont_norm /sum(cont_norm)

    exp_err_up = []
    exp_err_down = []
    for x in exp_norm:
        sd = np.sqrt(x*(1-x))
        err = sd/np.sqrt(len(exp))
        err = 1.96*err
        exp_err_up.append(err)
        if err > x:
            err = x
        exp_err_down.append(err)

    cont_err_up = []
    cont_err_down = []
    for x in cont_norm:
        sd = np.sqrt(x*(1-x))
        err = sd/np.sqrt(len(cont))
        err = 1.96*err
        cont_err_up.append(err)
        if err > x:
            err = x
        cont_err_down.append(err)

    dif = abs(bins[0]-bins[1])/2
    pos = []
    for x in bins[0:len(bins)-1]:
        pos.append(x + dif)

    plt.subplot(1,3,3).cla()
    plt.bar( pos, exp_norm, yerr=[exp_err_down, exp_err_up],  width= dif*0.7, align='edge', label=cond1, capsize=4, error_kw=dict(lw=1))
    plt.bar( pos, cont_norm, yerr=[cont_err_down, cont_err_up], width= dif*-0.7, align='edge', label=cond2, capsize=4, error_kw=dict(lw=1))

    plt.xlim((min(bins), max(bins)))
    plt.xticks(bins)
    plt.legend()


# takes arrary and number of bins and suggests bin distribution
def getBins(arr, number):
    # extend min max
    mini = np.floor(min(arr))
    maxi = np.ceil(max(arr))

    # get scale differences
    diff = abs(maxi - mini)
    diff_scale =  number * (np.ceil(diff/number) % diff)
    offset = (diff_scale - diff)

    # prject scale differences on real scale
    if (offset % 2) == 0:
        mini_scale = mini - offset/2
        maxi_scale = maxi + offset/2
    else:
        mini_scale = mini - np.floor(offset/2)
        maxi_scale = maxi + np.floor(offset/2) + 1

    # suggest bins
    step = diff_scale/number
    bins = np.arange(mini_scale, maxi_scale+1, step)
    pos = np.arange(mini_scale+step/2, maxi_scale, step)

    print(min(arr))
    print(max(arr))
    print(bins)
    print(pos)

    return bins
