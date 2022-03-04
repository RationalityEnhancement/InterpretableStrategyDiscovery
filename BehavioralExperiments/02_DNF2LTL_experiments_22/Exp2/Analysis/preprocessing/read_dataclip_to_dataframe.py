import json
from datetime import datetime
import pandas as pd
import numpy as np
import math

# ---------HELPER-----------------------
# -------------------------------------
def reduceNodeString(string):
    while string.find('.') != -1:
        index = string.find('.')
        string = string[:index] + string[index+3:]
    return string

def getMapping(dataclip, subject, timeline = None):

    if timeline is None:
        timeline = eval(json.loads(dataclip.iloc[subject]['datastring'])['questiondata']['global_timeline'])
        return getMapping(dataclip, subject, timeline)

    internal_node_ids = []
    descriptions = []
    keys = []

    for trial in json.loads(dataclip.iloc[subject]['datastring'])['data']:

        node_id = trial['trialdata']['internal_node_id']
        key = reduceNodeString(node_id)

        t = timeline
        for c in key:
            try:
                t = t[int(c)]
            except:
                t = 'undefined'

        internal_node_ids.append( node_id )
        descriptions.append(t)
        keys.append( key)

    trial_df = pd.DataFrame(list(zip(keys, descriptions, internal_node_ids)),
               columns =['Key', 'Des', 'Id'])

    return trial_df

# defines initial dataframe
def createDataframe(dataclip, getParticipantInfo):
    if dataclip.shape[0] == 0:
        return pd.DataFrame([], columns = [])

    df = pd.DataFrame([], columns = getParticipantInfo(dataclip, 0).keys())
    for i in range(len(dataclip)):
        df = df.append(getParticipantInfo(dataclip, i), ignore_index=True)
    return df

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


# --------- CUSTOM FUNCTION -----------------------
# -------------------------------------
# takes data_dict as dataclip and a subject index: extracts all relevant data for that partcipant


def getParticipantInfoExp1B(dataclip, subject):

    data = {
           'WorkerId' : '-',
           'hitId': '-',
           'assignmentId': '-',
           'status': '-',
           'dropoutIndex': '-',
           'condition': '-',
           'conditionType': '-',
           'age': '-',
           'gender': '-',
           'naive': '-',
           'feedback': '-',
           'totalTime': '-',
           'firstBlock': '-',
           'exclusion': '',
           'bonus': '-',
           'RT_bonus': '-',
           'MG_bonus': '-',
           'RT_attemptsQuiz': 0,
           'MG_attemptsQuiz': 0,
           'RT_trials': [],
           'MG_trials': [],
           'datastring': '-'}

    # main variables
    data['WorkerId'] = dataclip.iloc[subject]['workerid']
    data['hitId'] = dataclip.iloc[subject]['hitid']
    data['assignmentId'] = dataclip.iloc[subject]['assignmentid']
    data['status'] = dataclip.iloc[subject]['status']
    data['condition'] = int(dataclip.iloc[subject]['cond'])

    # compute total time
    begin_hit = dataclip.iloc[subject]['beginhit']
    begin_exp = dataclip.iloc[subject]['beginexp']
    end_hit = dataclip.iloc[subject]['endhit']

    if end_hit is not None and isinstance(end_hit, str):
        time_format = '%Y-%m-%d %H:%M:%S.%f'
        total_time = datetime.strptime(end_hit, time_format) - datetime.strptime(begin_hit, time_format)
        data['totalTime'] = round(total_time.seconds/60,1)

    # extract experiment content
    datastring = json.loads(dataclip.iloc[subject]['datastring'])
    if datastring is None:
        print("Aborted Experiment: No trial data")
        data['dropoutIndex'] = 0
    else:

        # condition
        data['dropoutIndex'] = 1
        data['datastring'] = datastring
        #data['condition'] = datastring['questiondata']['condition']
        data['conditionType'] = 'instructions' if data['condition'] in [2,3] else 'control'
        data['firstBlock'] = 'roadtrip' if data['condition'] in [1,3] else 'mortgage'

        # extract exclusion
        try:
            data['exclusion'] = datastring['questiondata']['exclusion']
        except:
            data['exclusion'] = '-'

        # extract bonus
        try:
            data['bonus'] = datastring['questiondata']['final_bonus']
            data['RT_bonus'] = datastring['questiondata']['roadtrip_bonus']
            data['MG_bonus'] = datastring['questiondata']['mortgage_bonus']
        except:
            data['bonus'] = 0

        # set trial codes
        first_quiz = ['011']
        second_quiz = ['041']

        if data['conditionType'] == 'control':
            first_task = ['0210', '0211', '0212', '0213', '0214', '0215', '0216', '0217']
            second_task = ['0510', '0511', '0512', '0513', '0514', '02515', '0516', '0517']
        else:
            first_task = ['0220', '0221', '0222', '0223', '0224', '0225', '0226', '0227']
            second_task = ['0520', '0521', '0522', '0523', '0524', '0525', '0526', '0527']

        if data['firstBlock'] == 'roadtrip':
            RT_codes = first_task
            RT_quiz = first_quiz
            MG_codes = second_task
            MG_quiz = second_quiz
        else:
            RT_codes = second_task
            RT_quiz = second_quiz
            MG_codes = first_task
            MG_quiz = first_quiz


        # loop through all trials and categorize  --------------------------------
        for trial in datastring['data']:

            trial_key = reduceNodeString(trial['trialdata']['internal_node_id'])


            # QUIZ
            if trial_key in RT_quiz:
                data['RT_attemptsQuiz'] += 1
                data['dropoutIndex'] = 2

            # QUIZ
            elif trial_key in MG_quiz:
                data['MG_attemptsQuiz'] += 1
                data['dropoutIndex'] = 3

            # Roatrip trials
            elif trial_key in RT_codes:
                data['RT_trials'].append(trial['trialdata'])

            # Roatrip trials
            elif trial_key in MG_codes:
                data['MG_trials'].append(trial['trialdata'])

            # FINAL SURVEY
            elif trial_key in ['07']:
                data['dropoutIndex'] = 4
                resp = trial['trialdata']['response']
                data['naive'] = resp['naive'] == 'No'
                data['age'] = resp['age']
                data['gender'] = resp['gender']

            elif trial_key in ['08']:
                resp = trial['trialdata']['response']
                data['feedback'] = resp['Q0']
                data['dropoutIndex'] = 5


    if len(data['RT_trials']) + len(data['MG_trials']) != 16 and not data['status'] == 6:
        data['exclusion'] = True

    return data
