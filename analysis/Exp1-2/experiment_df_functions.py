import json
from datetime import datetime
import pandas as pd
import numpy as np

def print_clicks(datastring):
    for trial in datastring['data']:
        trial_key = reduceNodeString(trial['trialdata']['internal_node_id'])

        if trial_key in ['0210', '0211', '0212', '0213', '0214', '0215', '0216', '0217', '0218', '0219']:
            print('State Rewards: ' + str([-1,10,20,30,40,50,60,70,80,90,10,11,12]))
            print('State Rewards: ' + str(trial['trialdata']['stateRewards']))
            print('Clicked: ' + str(trial['trialdata']['queries']['click']['state']['target']))
            print(' ')
            
            
# takes data_dict as dictionary and a subject index: extracts all relevant data for that partcipant
def getParticipantInfo3(dictionary, subject):

    # retrieve main infos on first level of dictionary
    begin_hit = dictionary[subject]['beginhit']
    begin_exp = dictionary[subject]['beginexp']
    end_hit = dictionary[subject]['endhit']
    worker_id = dictionary[subject]['workerid']
    hit_id = dictionary[subject]['hitid']
    assignment_id = dictionary[subject]['assignmentid']
    condition = dictionary[subject]['cond']
    datastring = dictionary[subject]['datastring']
    status = dictionary[subject]['status']
    total_time = '-'

    # compute total time
    time_format = '%Y-%m-%d %H:%M:%S.%f'
    if end_hit is not None:
        total_time = datetime.strptime(end_hit, time_format) - datetime.strptime(begin_hit, time_format)
        total_time = total_time.seconds


    # define custom variables to extract from expereiment with default value
    condition_type = '-'
    attempts = 0
    genEdu = '-'
    comEdu = '-'
    age = '-'
    gender = '-'
    qUnd  = '-'
    qDes = '-'
    feedback = '-'
    cheat_trials  = '-'
    number_clicks = []
    ML_trials = []
    trial_time = []
    trial_time_mean = '-'

    if dictionary[subject]['datastring'] is None:
        print("Aborted Experiment: No trial data")
    else:
        # extract condition info
        condition_type = datastring['questiondata']['condition_type']


        # loop thorugh all trials and categorize
        for trial in datastring['data']:
            trial_key = reduceNodeString(trial['trialdata']['internal_node_id'])


            # --------0-------
            if condition == 0 or condition == 1:

                if trial_key in ['013']:
                    attempts += 1

                if trial_key in ['030']:
                    try:
                        resp = eval(trial['trialdata']['responses'])
                        qUnd = resp['Q0']
                        qDes = resp['Q1']
                        age = resp['Q2']
                        gender = resp['Q3']
                    except:
                        pass


             # ---------1------
            if condition == 2:

                if trial_key in ['011']:
                    attempts += 1

                if trial_key in ['030']:
                    try:
                        resp = eval(trial['trialdata']['responses'])
                        age = resp['Q0']
                        gender = resp['Q1']
                    except:
                        pass


            # shared
            if trial_key in ['0210', '0211', '0212', '0213', '0214', '0215', '0216', '0217', '0218', '0219']:
                ML_trials.append(trial)
                trial_time.append(int(trial['trialdata']['trialTime']/1000))

            if trial_key in ['031']:
                try:
                    genEdu = eval(trial['trialdata']['responses'])[0]
                    comEdu = eval(trial['trialdata']['responses'])[1]
                except:
                    pass

            if trial_key in ['032']:
                try:
                    feedback = eval(trial['trialdata']['responses'])['Q0']
                except:
                    pass


    # test cheat trials
    if len(ML_trials) < 10:
        print("Aborted Experiment: Less than 10 testing trials")
        cheat_trials = 10-len(ML_trials)
    else:

        # extract cheat trials
        threshold_clicks = 1
        cheat_trials = 0
        trial_time_mean = sum(trial_time)/len(trial_time)

        for trial in ML_trials:
            if(trial['trialdata']['block'] == "testing"):
                num_clicks = len(trial['trialdata']['queries']['click']['state']['target'])
                number_clicks.append(num_clicks)

                if(num_clicks < threshold_clicks):
                    cheat_trials += 1


    return {'WorkerId' : worker_id,
           'hitId': hit_id,
            'assignmentId': assignment_id,
            'condition': condition,
            'condition_type': condition_type,
           'beginHit': begin_hit,
           'beginExp': begin_exp,
           'endHit': end_hit,
           'totalTime': total_time,
           'attempts': attempts,
           'genEdu': genEdu ,
           'comEdu': comEdu,
           'age': age,
           'gender': gender,
           'qUnd': qUnd,
           'qDes': qDes,
           'feedback': feedback,
           'cheatTrials': cheat_trials,
           'status': status,
           'numberClicks': number_clicks,
           'trialTimes': trial_time,
           'trialTimeMean': trial_time_mean,
           'datastring': datastring}



# takes data_dict as dictionary and a subject index: extracts all relevant data for that partcipant
def getParticipantInfo(dictionary, subject):

    # retrieve main infos on first level of dictionary
    begin_hit = dictionary[subject]['beginhit']
    begin_exp = dictionary[subject]['beginexp']
    end_hit = dictionary[subject]['endhit']
    worker_id = dictionary[subject]['workerid']
    hit_id = dictionary[subject]['hitid']
    assignment_id = dictionary[subject]['assignmentid']
    condition = dictionary[subject]['cond']
    datastring = dictionary[subject]['datastring']
    status = dictionary[subject]['status']
    total_time = '-'

    # compute total time
    time_format = '%Y-%m-%d %H:%M:%S.%f'
    if end_hit is not None:
        total_time = datetime.strptime(end_hit, time_format) - datetime.strptime(begin_hit, time_format)
        total_time = total_time.seconds


    # define custom variables to extract from expereiment with default value
    condition_type = '-'
    attempts = '-'
    genEdu = '-'
    comEdu = '-'
    age = '-'
    gender = '-'
    qUnd  = '-'
    qDes = '-'
    feedback = '-'
    cheat_trials  = '-'
    number_clicks = []

    rews_exp = []



    if dictionary[subject]['datastring'] is None:
        print("Aborted Experiment: No trial data")
    else:

        # scan for trial types that are of interest e.g. mouselab trials, survery multiple choice, survey text trials
        ML_trials = []
        SMC_trials = []
        ST_trials = []

        # loop thorugh all trials and categorize
        data = datastring['data']
        for trial in data:
            trial_type = trial['trialdata']['trial_type']

            if trial_type == 'mouselab-mdp':
                if trial['trialdata']['block'] == 'testing':
                    ML_trials.append(trial)

            elif trial_type == 'survey-multi-choice' :
                SMC_trials.append(trial)

            elif trial_type == 'survey-text':
                ST_trials.append(trial)


        # extract condition info
        condition_type = datastring['questiondata']['condition_type']

        if len(ML_trials) < 10:
            print("Aborted Experiment: Less than 10 testing trials")
            cheat_trials = 10-len(ML_trials)
        else:

            # extract cheat trials
            threshold_clicks = 1
            cheat_trials = 0
            for trial in ML_trials:
                if(trial['trialdata']['block'] == "testing"):
                    num_clicks = len(trial['trialdata']['queries']['click']['state']['target'])
                    number_clicks.append(num_clicks)

                    if(num_clicks < threshold_clicks):
                        cheat_trials += 1

                    state = [0,0,0,0,0,0,0,0,0,0,0,0,0]
                    for c in trial['trialdata']['queries']['click']['state']['target']:
                        state[int(c)] = int(trial['trialdata']['stateRewards'][int(c)])
                    rews_exp.append(getExpectedReward(state) - len(trial['trialdata']['queries']['click']['state']['target']))



            #    print("-----how to acccess mouselab data-----")
            #    print('Block: ' + x['trialdata']['block'] )
            #    print('State Rewards: ' + str(x['trialdata']['stateRewards']))
            #    print('Clicked: ' + str(x['trialdata']['queries']['click']['state']['target']))


            # extract attempts to pass quiz
            attempts = len(SMC_trials) - 1


            #  extract Survey Multi Choice answers
            try:
                genEdu = eval(SMC_trials[len(SMC_trials)-1]['trialdata']['responses'])[0]
            except:
                pass
            try:
                comEdu = eval(SMC_trials[len(SMC_trials)-1]['trialdata']['responses'])[1]
            except:
                pass


            # Extract Survey Text answers (conditions have different questions)
            # First questionnaire
            resp = '-'
            try:
                resp = eval(ST_trials[0]['trialdata']['responses'])
            except:
                pass

            if resp != '-':
                if(condition == 1):
                    questions = ['What is your age?', 'Which gender do you identify with?']

                    age = resp['Q0']
                    gender = resp['Q1']
                    qUnd = '-'
                    qDes = '-'

                else:
                    questions = ['Did you understand the strategy that the tree was conveying?', 'If you were describe it to a friend, what would you say?', 'What is your age?', 'Which gender do you identify with?']

                    qUnd = resp['Q0']
                    qDes = resp['Q1']
                    age = resp['Q2']
                    gender = resp['Q3']


            # Second questionnaire, feedback
            resp = '-'
            try:
                resp = eval(ST_trials[1]['trialdata']['responses'])
            except:
                pass

            if resp != '-':
                feedback = resp['Q0']


    return {'WorkerId' : worker_id,
           'hitId': hit_id,
            'assignmentId': assignment_id,
            'condition': condition,
            'condition_type': condition_type,
           'beginHit': begin_hit,
           'beginExp': begin_exp,
           'endHit': end_hit,
           'totalTime': total_time,
           'attempts': attempts,
           'genEdu': genEdu ,
           'comEdu': comEdu,
           'age': age,
           'gender': gender,
           'qUnd': qUnd,
           'qDes': qDes,
           'feedback': feedback,
           'cheatTrials': cheat_trials,
           'status': status,
           'numberClicks': number_clicks,
           'rews_exp': rews_exp,
           'rews_exp_total': np.sum(rews_exp),
           'rews_exp_mean': np.mean(rews_exp),
           'datastring': datastring}



# ---------HELPER-----------------------
# -------------------------------------

def getExpectedReward(state):
    m = -10000
    paths = [[1,2,3], [1,2,4], [5,6,7], [5,6,8], [9,10,11], [9,10,12]]
    for path in paths:
        rew = 0
        for s in path:
            rew += state[s]
        m = max(m, rew)
    return m

# defines initial dataframe
def makeDataframe(data_dict):
    if len(data_dict) == 0:
        return pd.DataFrame([], columns = [])

    df = pd.DataFrame([], columns = getParticipantInfo(data_dict, 0).keys())

    for i in range(len(data_dict)):
        df = df.append(getParticipantInfo(data_dict, i), ignore_index=True)
    return df

def pretty_print(obj):
    print(json.dumps(obj, indent=4, sort_keys=False))

def addColumn2Df(df, name, values):
    df[name] = values
    return df

def df2listColumn(df):
    res = []
    for i in list(df):
        res.append(df[i].tolist())
    return res

def df2listRow(df):
    res = []
    for index, row in df.iterrows():
        res.append(df.iloc[index])
    return res

def joinColumnToDf(df, identifier, identifier_name, values, values_name):
    other = pd.DataFrame({identifier_name: identifier, values_name: values})
    new_df = df.set_index(identifier_name).join(other.set_index(identifier_name))

    new_df.reset_index(drop=False, inplace=True)
    return new_df

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


def reduceNodeString(string):
    while string.find('.') != -1:
        index = string.find('.')
        string = string[:index] + string[index+3:]
    return string

def getMapping(data_dict, subject, timeline = None):

    if timeline is None:
        timeline = json.loads(data_dict[subject]['datastring']['questiondata']['global_timeline'])
        return getMapping(data_dict, subject, timeline)

    internal_node_ids = []
    descriptions = []
    keys = []

    for trial in data_dict[subject]['datastring']['data']:

        node_id = trial['trialdata']['internal_node_id']
        key = reduceNodeString(node_id)

        t = timeline
        for c in key:
            t = t[int(c)]

        internal_node_ids.append( node_id )
        descriptions.append(t)
        keys.append( key)

    trial_df = pd.DataFrame(list(zip(keys, descriptions, internal_node_ids)),
               columns =['Key', 'Des', 'Id'])

    return trial_df

# takes df (with totalTime in seconds and Bonus column), base pay and desired minimum wage per Hour
# returns array with delta payments raising everyone to minimum wage
def getDeltaBonus(df, base_pay, min_wage_ph, total_bonus = False):

    def print_wage_stats(wage):
        print('/n /n')
        print('Wage per Hour:')
        print('Min: {}'.format(wage.min()))
        print('Max: {}'.format(wage.max()))
        print('Mean: {}'.format(wage.mean()))


    # compute average per hour
    minutes = df.totalTime /60
    factor = 60 / minutes
    if total_bonus:
        bonus = df.total_bonus
    else:
        bonus = df.Bonus
    wage_ph = factor * (bonus + base_pay)
    print_wage_stats(wage_ph)

    # compute delta per minute
    delta = (min_wage_ph - wage_ph) / 60
    delta_bonus = minutes * delta
    delta_bonus[delta_bonus < 0] = 0

    new_wage_ph = factor * (bonus + base_pay + delta_bonus)
    print_wage_stats(new_wage_ph)

    return delta_bonus, wage_ph, new_wage_ph


# writes a python file with input for bouns_computation.py
def exportBonus(df, total_bonus = False, additional_ids = None, additional_bonus = None):
    with open("bonus_input_file.py", "w") as output:

        output.write('assignmentIDs = ')
        output.write(str(df.assignmentId.tolist()) + '\n \n')

        output.write('workerIDs = ')
        output.write(str(df.WorkerId.tolist()) + '\n \n')

        output.write('bonuses = ')
        if total_bonus:
            output.write(str(df.total_bonus.tolist()) + '\n \n')
        else:
            output.write(str(df.bonus.tolist()) + '\n \n')

        output.write('hitID = ')
        output.write("'" + df.hitId[0] + "'" + '\n \n')

        if not additional_ids is None:
            output.write('# Early Quitter: ' + '\n')
            for ids,bon in zip(additional_ids, additional_bonus):
                output.write("# " + str(ids) + ': ' + str(bon) + '\n')
