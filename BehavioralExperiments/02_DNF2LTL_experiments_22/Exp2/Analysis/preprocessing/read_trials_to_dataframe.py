import json
from datetime import datetime
import pandas as pd
import numpy as np
import math

# defines initial dataframe
def createTrialframe(trials, pids, getTrialInfo, task):
    '''
    input:
        - list of lists of trials
        - list of pids
        - function to extract trialinfos
    output:
        - pandas dataframe with one row per single trial
    '''
    if trials.shape[0] == 0:
        return pd.DataFrame([], columns = [])

    df = pd.DataFrame([], columns = getTrialInfo(eval(trials[0])[0], pids[0], 0, task).keys())
    for p in range(len(pids)):
        t = eval(trials[p])
        for i in range(len(t)):
            df = df.append(getTrialInfo(t[i], pids[p], i, task), ignore_index=True)
    return df


def getRoadtripInfo(trial, pid, index, task):
    '''
    input:
        - roadtirp trialinfo
        - participant pid
        - index of trial
    output:
        - dictionary with releveant extracted informations
    '''

    def getReward(key):
        return trial['nodes'][str(key)]['reward']

    data = {
            'pid': pid,
            'task': task,
            'trial': index,
            'numberClicks': len(trial['clicks']),
            'clicks': trial['clicks'],
            'score': trial['trial_score'],
            'FSQ': getFSQ(trial['clicks'], trial['end_nodes']),
            'CA': getClickAgreement(trial['clicks'], trial['end_nodes'], 20, getReward)}

    return data

def getMortageInfo(trial, pid, index, task):
    '''
    input:
        - mortgage trialinfo
        - participant pid
        - index of trial
    output:
        - dictionary with releveant extracted informations
    '''

    def getReward(key):
        #if( len(trial['all_node_values']) <= int(key)):
        #    return 100

        entry = trial['all_node_values'][int(key)]
        return float(entry[0:len(entry)-1])

    data = {
            'pid': pid,
            'trial': index,
            'task': task,
            'numberClicks': len(trial['response_interest_rate']),
            'clicks': trial['response_interest_rate'],
            'score': int(trial['trial_score'] == 10),
            'FSQ': getFSQ(trial['response_interest_rate'], [2,5,8]),
            'CA': getClickAgreement(trial['response_interest_rate'], [2,5,8], 0, getReward)}

    return data

def getFSQ(clicks, end_nodes):
    '''
    input:
         - clicks the participant made
         - end nodes
    output:
        - FSQ in [0,1]
    '''

    # counting variables
    good_clicks = 0
    bad_clicks = 0

    # number of first k clicks to consider
    k = len(end_nodes)
    for i in range(k):

        # stop if
        if(len(clicks) <= i):
            break

        if(clicks[i] in end_nodes):
            good_clicks += 1

        else:
            bad_clicks += 1

    # FSQ calculation
    return good_clicks/max(1, (good_clicks + bad_clicks))


def getClickAgreement(clicks, end_nodes, best_value, getReward):
    '''
    input:
         - clicks the participant made
         - end nodes
         - stopping criterion
         - function to extract reward
    output:
        - Click Agreement in [0,1]
    '''

    # counting variables
    good_clicks = 0
    bad_clicks = 0
    best_value_found = False

    #print(clicks)
    #print(end_nodes)

    # evaluate clicks performed
    for c in clicks:

        if(c in end_nodes and not best_value_found):
            good_clicks += 1

            if(getReward(c) <= best_value):

                # best value found --> no more good clicks
                best_value_found = True

        else:
            bad_clicks += 1

    # early termination --> calculate how many more clicks to expect
    if(not best_value_found):
        remaining_end_nodes = len(end_nodes) - good_clicks

        if(remaining_end_nodes > 0):

            # Expecation of number of clicks until finding best value
            expected_number_of_additional_clicks = np.mean(np.arange(1,remaining_end_nodes+1))
            bad_clicks += max(expected_number_of_additional_clicks - bad_clicks, 0)

    # final calculation
    #print(good_clicks)
    #print(bad_clicks)
    #print()

    return good_clicks/max(1, (good_clicks + bad_clicks))
