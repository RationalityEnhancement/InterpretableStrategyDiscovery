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
            t = t[int(c)]

        internal_node_ids.append( node_id )
        descriptions.append(t)
        keys.append( key)

    trial_df = pd.DataFrame(list(zip(keys, descriptions, internal_node_ids)),
               columns =['Key', 'Des', 'Id'])

    return trial_df

# defines initial dataframe
def makeDataframe(dataclip, getParticipantInfo):
    if dataclip.shape[0] == 0:
        return pd.DataFrame([], columns = [])

    df = pd.DataFrame([], columns = getParticipantInfo(dataclip, 0).keys())

    for i in range(dataclip.shape[0]):
        df = df.append(getParticipantInfo(dataclip, i), ignore_index=True)
    return df

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

# takes df (with totalTime in seconds and Bonus column), base pay and desired minimum wage per Hour
# returns array with delta payments raising everyone to minimum wage
def getDeltaBonus(df, base_pay, min_wage_ph, max_bonus= 1000, total_bonus = False):

    def print_wage_stats(wage):
        print('/n /n')
        print('Wage per Hour:')
        print('Min: {}'.format(wage.min()))
        print('Max: {}'.format(wage.max()))
        print('Mean: {}'.format(wage.mean()))

    if sum([isinstance(x, str) for x in df.totalTime ]) > 0:
        print('Error: Strings in totalTime')
        #isinstance(x, str) for x in df.totalTime ]

    minutes = np.array([df.totalTime.mean() if np.isnan(x) else x for x in df.totalTime])

    # compute average per hour

    #minutes = df.totalTime
    factor = 60 / minutes
    if total_bonus:
        bonus = df.total_bonus
    else:
        bonus = df.bonus
    wage_ph = factor * (bonus + base_pay)
    print_wage_stats(wage_ph)

    # compute delta per minute
    delta = (min_wage_ph - wage_ph) / 60
    delta_bonus = minutes * delta
    delta_bonus[delta_bonus < 0] = 0

    for i in range(len(delta_bonus)):
        if delta_bonus[i] + bonus[i] > max_bonus:
            delta_bonus[i] = max_bonus-bonus[i]

    new_wage_ph = factor * (bonus + base_pay + delta_bonus)
    print_wage_stats(new_wage_ph)

    return delta_bonus, wage_ph, new_wage_ph


# writes a python file with input for bouns_computation.py
def exportBonus(df, df_all, total_bonus = False, additional_ids = None, additional_bonus = None, exp_version = ''):
    with open('data/Exp'+exp_version+'/input_exp'+exp_version+'.py', "w") as output:

        output.write('assignmentIDs = ')
        output.write(str(df.assignmentId.tolist()) + '\n \n')

        output.write('workerIDs = ')
        output.write(str(df.WorkerId.tolist()) + '\n \n')

        output.write('bonuses = ')
        if total_bonus:
            output.write(str(df.total_bonus.tolist()) + '\n \n')
        else:
            output.write(str(df.bonus.tolist()) + '\n \n')

        output.write('hitIDs = ')
        output.write(str(df.hitId.unique().tolist())  + '\n \n')

        output.write('workerIDsQualification = ')
        output.write(str(df_all.WorkerId.tolist()) + '\n \n')

        if not additional_ids is None:
            output.write('# Early Quitter: ' + '\n')
            for ids,bon in zip(additional_ids, additional_bonus):
                output.write("# " + str(ids) + ': ' + str(bon) + '\n')

def calcExpectedReward(state):
    m = -10000
    paths = [[1,2,3], [1,2,4], [5,6,7], [5,6,8], [9,10,11], [9,10,12]]
    for path in paths:
        rew = 0
        for s in path:
            rew += state[s]
        m = max(m, rew)
    return m

def getExpectedScore(trial):
    state = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    sr = trial['trialdata']['stateRewards']
    clicks = trial['trialdata']['queries']['click']['state']['target']

    for click in clicks:
        state[int(click)] = sr[int(click)]

    return calcExpectedReward(state)-len(clicks)



# --------- CUSTOM FUNCTION -----------------------
# -------------------------------------
# takes data_dict as dictionary and a subject index: extracts all relevant data for that partcipant

def getParticipantInfoExp2(dataclip, subject):

    # retrieve main infos on first level of dictionary
    begin_hit = dataclip.iloc[subject]['beginhit']
    begin_exp = dataclip.iloc[subject]['beginexp']
    end_hit = dataclip.iloc[subject]['endhit']
    worker_id = dataclip.iloc[subject]['workerid']
    hit_id = dataclip.iloc[subject]['hitid']
    assignment_id = dataclip.iloc[subject]['assignmentid']
    condition = dataclip.iloc[subject]['cond']
    datastring = json.loads(dataclip.iloc[subject]['datastring'])
    status = dataclip.iloc[subject]['status']
    total_time = np.NaN

    # define custom variables to extract from expereiment with default value
    condition_type = '-'
    attempts_quiz = 0
    bonus = 0

    # stats
    age = '-'
    gender = '-'
    feedback = '-'
    qConfusing = '-'
    prompts = []
    qFollowStrategy = '-'

    # training
    ML_trials = []
    ML_cheat_trials = 0
    ML_clicks_number = []

    # testing
    RT_trials = []
    RT_clicks_number = []
    RT_cheat_trials = 0
    RT_scores = []
    RT_scores_exp = []

    time_tracking = [0,0] #instructions, endTesting


    if datastring is None:
        print("Aborted Experiment: No trial data")
    else:
        # extract condition info
        condition_type = datastring['questiondata']['condition_type']
        #condition = int(datastring['questiondata']['condition'])

        try:
            bonus = datastring['questiondata']['final_bonus']
        except:
            pass

        # loop thorugh all trials and categorize
        for trial in datastring['data']:
            trial_key = reduceNodeString(trial['trialdata']['internal_node_id'])
            prompts_code = []
            ML_code = []


            RT_code = ['0220', '0221', '0222', '0223', '0224','0225','0226','0227']
            quest1_code = ['030']
            quest2_code = ['031']

            #if condition > 0:
            ML_code = ['0140', '0141', '0142', '0143', '0144','0145','0146','0147','0148','0149']

            if trial_key in ['0200']:
                time_tracking[0] = round(trial['trialdata']['time_elapsed']/60000,1)

            if trial_key in ['0201']:
                attempts_quiz += 1

            # mouselab test trials
            if trial_key in ML_code:
                ML_trials.append(trial['trialdata'])

                # clicks
                num_clicks = len(trial['trialdata']['queries']['click']['state']['target'])
                ML_clicks_number.append(num_clicks)

                if(num_clicks < 1):
                    ML_cheat_trials += 1


            # mouselab test trials
            if trial_key in RT_code:
                if 'clicks' not in trial['trialdata']:
                    status = 2
                else:
                    RT_trials.append(trial['trialdata'])

                    # clicks
                    num_clicks = len(trial['trialdata']['clicks'])

                    RT_clicks_number.append(num_clicks)

                    score = trial['trialdata']['cost_total']
                    RT_scores.append(score)

                    score_exp = trial['trialdata']['cost_expected']
                    RT_scores_exp.append(score_exp)

                    if(num_clicks < 1):
                        RT_cheat_trials += 1


            # questionaire 1
            if trial_key in quest1_code:
                time_tracking[1] = round(trial['trialdata']['time_elapsed']/60000,1)
                try:
                    resp = eval(trial['trialdata']['responses'])
                    qConfusing = resp['Q0']
                    age = resp['Q1']
                    gender = resp['Q2']

                    if condition > 0:
                        qFollowStrategy = resp['Q3']
                except:
                    pass

            # questionaire 2
            if trial_key in quest2_code:
                total_time = round(trial['trialdata']['time_elapsed']/60000,1)
                try:
                    feedback = eval(trial['trialdata']['responses'])['Q0']
                except:
                    pass

    return {
    'WorkerId' : worker_id,
    'hitId': hit_id,
    'assignmentId': assignment_id,
    'condition': condition,
    'conditionType': condition_type,
    'status': status,
    #'bonus': round(bonus,2),
    #'beginHit': begin_hit,
    #'beginExp': begin_exp,
    #'endHit': end_hit,
    #'timeTracking': time_tracking,
    'totalTime': total_time,
    'attemptsQuiz1': attempts_quiz,
    'gender': gender,
    'age': age,
    #'qConfusing': qConfusing,
    #'qFollowStrategy': qFollowStrategy,
    #'feedback': feedback,
    'MLcheatTrials': ML_cheat_trials,
    #'MLclicksNumber': ML_clicks_number,
    #'RTcheatTrials': RT_cheat_trials,
    #'RTclicksNumber': RT_clicks_number,
    #'RTclicksNumberMean': np.mean(RT_clicks_number),
    'RTscores': RT_scores,
    'RTscoresMean': np.round(np.mean(RT_scores), 4),
    #'RTscoresExp': RT_scores_exp,
    #'RTscoresExpMean': np.round(np.mean(RT_scores_exp), 4),
    #'prompts' : prompts,
    'trainingTrials': ML_trials,
    'testingTrials': RT_trials,
    'datastring': datastring
    }


'''
# --------- MAIN -----------------------
# -------------------------------------

# ----- IMPORT
# import data as json loaded from postgre database
with open('data/dataclipsExp11.json', 'r') as handle:
    data = json.load(handle)

data_dict = json2dict(data)
index_cond0 = 10 # index of data_dict for a full expererimental trial
index_cond1 = 25 # index of data_dict for a full control trial


# -----  MAPPINGS: shows an identifying key for each trial

Include this in experiment.js, after the definition of experiment_timeline.

  flatten_timeline = function(timeline){
    var global_timeline = [];

    for(var i in timeline){
      t = timeline[i];

      if(t.timeline !== undefined){
        //recursive for sub timelines
        global_timeline.push( flatten_timeline( t.timeline ));
      } else {
        // its a real block
        if(t.type !== undefined){
          info = t.type;

          if(t.questions !== undefined){
            info = info + ' : ' + t.questions.toString();
          }
          global_timeline.push( info);

        } else if (t.trial_id !== undefined){
          global_timeline.push( 'Mouselab : ' + t.trial_id)
        }

      }
    }
    global_timeline = [global_timeline.flat(1)];
    return(global_timeline);
  }
  psiturk.recordUnstructuredData('global_timeline', JSON.stringify(flatten_timeline(experiment_timeline)) );
  //console.log( JSON.stringify(flatten_timeline(experiment_timeline)) );


print('Experimental:')
print(getMapping(data_dict, index_cond0))
print('')
print('Control:')
print(getMapping(data_dict, index_cond1))
print('')


# ----- MODIFY getParticpantInfo to extract all relevant infromations
pretty_print(getParticipantInfoExp11(data_dict, index_cond0))
pretty_print(getParticipantInfoExp11(data_dict, index_cond1))
print('')


# ----- CREATE dataframe
dataframe = makeDataframe(data_dict, getParticipantInfoExp11)
print(len(dataframe))
print('')

# ---- REMOVE Early Quitter
dataframe = dataframe[dataframe.endHit != '-'].reset_index(drop=True, inplace=False)
print(len(dataframe))
print('')

# ---- computes delta to reach min wage for every one
#delta_bonus = getDeltaBonus(dataframe , basepay, minwage)
delta_bonus, wage_ph, new_wage_ph = getDeltaBonus(dataframe , 0.5, 4)
dataframe['delta_bonus'] = delta_bonus.tolist()
dataframe["total_bonus"] = dataframe['bonus'] + dataframe['delta_bonus']

# --- export bonus for bonus_computation.py
exportBonus(dataframe, total_bonus = False)
dataframe.to_csv('data/dataframe.csv')

'''
