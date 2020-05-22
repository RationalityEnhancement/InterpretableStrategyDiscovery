import json
from datetime import datetime
import pandas as pd
import numpy as np


# ---------HELPER-----------------------
# -------------------------------------

# defines initial dataframe
def makeDataframe(data_dict, getParticipantInfo):
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

def print_clicks(datastring):
    for trial in datastring['data']:
        trial_key = reduceNodeString(trial['trialdata']['internal_node_id'])

        if trial_key in ['0210', '0211', '0212', '0213', '0214', '0215', '0216', '0217', '0218', '0219']:
            print('State Rewards: ' + str([-1,10,20,30,40,50,60,70,80,90,10,11,12]))
            print('State Rewards: ' + str(trial['trialdata']['stateRewards']))
            print('Clicked: ' + str(trial['trialdata']['queries']['click']['state']['target']))
            print(' ')

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
    else:
        # compute average per hour
        minutes = df.totalTime /60
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
def exportBonus(df, total_bonus = False, additional_ids = None, additional_bonus = None):
    with open("data/bonus_input_file.py", "w") as output:

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

def getExpectedReward(state):
    m = -10000
    paths = [[1,2,3], [1,2,4], [5,6,7], [5,6,8], [9,10,11], [9,10,12]]
    for path in paths:
        rew = 0
        for s in path:
            rew += state[s]
        m = max(m, rew)
    return m

# --------- CUSTOM FUNCTION -----------------------
# -------------------------------------
# takes data_dict as dictionary and a subject index: extracts all relevant data for that partcipant


# takes data_dict as dictionary and a subject index: extracts all relevant data for that partcipant
def getParticipantInfoExp3(dictionary, subject):

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
    if (end_hit is not None) and (begin_exp is not None):
        total_time = datetime.strptime(end_hit, time_format) - datetime.strptime(begin_hit, time_format)
        total_time = total_time.seconds
    else:
        end_hit = '-'

    # define custom variables to extract from expereiment with default value
    condition_type = '-'
    attempts_quiz = 0
    attempts_quiz2 = 0
    attempts_practice = 0
    genEdu = '-'
    comEdu = '-'
    age = '-'
    gender = '-'
    qUnd  = '-'
    qConf = '-'
    qNotUnd = '-'
    qYouAsk = '-'
    feedback = '-'
    cheat_trials  = 0
    number_clicks = []
    clicks = []
    preselect = []
    flow_index = []
    flow_answer = []
    flow_correct = []
    flow_index_count = 0
    flow_correct_count = 0

    flow_correct_perc_training = []
    flow_correct_perc_testing = []
    trial_flow_all_train = [0, 0, 0]
    trial_flow_cor_train = [0, 0, 0]
    trial_flow_all_test = [0, 0, 0]
    trial_flow_cor_test = [0, 0, 0]
    flowchart = []

    trial_time_finish = []
    trial_time = []
    trial_time_mean = '-'
    count_ML_trials = 0
    ML_test_trials = []
    ML_add_trials = []

    rews = []
    rews_norm = []
    rews_good = []
    rews_exp = []

    time_tracking = [0,0,0] #quiz1, startTest, endTest
    scores = []

    num_correct_clicks = 0
    num_incorrect_clicks = 0

    if dictionary[subject]['datastring'] is None:
        print("Aborted Experiment: No trial data")
    else:
        # extract condition info
        condition_type = datastring['questiondata']['condition_type']
        condition = int(datastring['questiondata']['condition'])


        # loop thorugh all trials and categorize
        for trial in datastring['data']:
            trial_key = reduceNodeString(trial['trialdata']['internal_node_id'])

            # --------0-------
            if condition == 0:

                # questionaire 1
                if trial_key in ['061']:
                    time_tracking[2] = round(trial['trialdata']['time_elapsed']/60000)
                    try:
                        resp = eval(trial['trialdata']['responses'])
                        qUnd = resp['Q0']
                        qConf = resp['Q1']
                        qNotUnd = resp['Q2']
                        qYouAsk = resp['Q3']
                        age = resp['Q4']
                        gender = resp['Q5']
                    except:
                        pass

             # ---------1------
            if condition == 1:

                # questionaire 1
                if trial_key in ['061']:
                    try:
                        resp = eval(trial['trialdata']['responses'])
                        age = resp['Q0']
                        gender = resp['Q1']
                    except:
                        pass

                if trial_key in ['0510','0511','0512','0513','0514','0515','0516','0517','0518','0519','05110','05111','05112','05113','05114','05115','05116','05117','05118','05119','05120','05121','05122','05123','05124','05125','05126','05127','05128','05129']:

                    scores.append(trial['trialdata']['score'])


            # --------  shared -----
            # quiz attempts
            if trial_key in ['011']:
                attempts_quiz += 1
                time_tracking[0] = round(trial['trialdata']['time_elapsed']/60000)

            # practice attempts
            if trial_key in ['02100']:
                attempts_practice += 1

            if trial_key in ['031']:
                attempts_quiz2 += 1

            if trial_key in ['0410', '0411']:
                ML_add_trials.append(trial)

            ML_test_code = ['0510','0511','0512','0513','0514','0515','0516','0517','0518','0519']

            # questionaire 2
            if trial_key in ['062']:
                time_tracking[2] = round(trial['trialdata']['time_elapsed']/60000)
                try:
                    genEdu = eval(trial['trialdata']['responses'])[0]
                    comEdu = eval(trial['trialdata']['responses'])[1]
                except:
                    pass

            # questionaire 3
            if trial_key in ['063']:
                try:
                    feedback = eval(trial['trialdata']['responses'])['Q0']
                except:
                    pass


            if trial_key in ML_test_code[0]:
                time_tracking[1] = round(trial['trialdata']['time_elapsed']/60000)

            # mouselab test trials
            if trial_key in ML_test_code:
                ML_test_trials.append(trial)

                num_clicks = len(trial['trialdata']['queries']['click']['state']['target'])
                number_clicks.append(num_clicks)
                if(num_clicks < 1):
                    cheat_trials += 1

                trial_time.append(int(trial['trialdata']['trialTime']/1000))
                trial_time_finish.append(int(trial['trialdata']['actionTimes'][2]/1000))
                count_ML_trials += 1


                # rewards
                opt_scores = [14.776, -3.986, 3.42, -2.095, -5.384, 13.735, 21.46, 18.205, 15.205, 26.102, -3.592, -11.34, 8.561, 26.037, 21.339, 19.175, 2.951, 13.56, -18.974, 20.408, -3.496, 9.59, 3.454, 21.29, 19.294, -3.159, 14.918, 23.958, 12.076, 15.415, 25.873, -10.239, 15.183, 24.272, -1.629, 18.691, 20.886, 8.473, 3.752, 9.89, 6.485, -2.548, 1.084, 9.414, 2.906, 24.687, -1.618, 8.037, 20.351, 20.932, 19.016, 9.662, 3.045, 13.356, -1.811, 15.837, 21.041, 4.507, 3.444, 19.261, 19.972, 3.042, 15.414, 9.903, 13.696, 9.349, 26.569, -2.07, 15.845, 3.474, 3.636, 1.235, 18.776, 21.794, -1.199, -8.318, 8.654, 4.562, 14.985, -7.203, 18.631, 3.853, 9.331, 3.061, 15.887, 2.743, 7.904, 3.733, -2.899, 4.0, -0.829, 13.798, 17.981, 3.388, 9.194, 8.742, 19.275, 1.526, 9.479, -8.667]

                good_scores = [0.451, -5.828, 5.01, 1.795, -9.253, 11.894, 20.966, 16.305, 15.239, 25.311, -2.425, -11.627, 9.818, 17.751, 21.416, 15.851, -0.3, -0.511, -16.819, 20.761, -9.621, 10.664, -4.096, 20.507, 18.805, -9.567, 4.779, 23.872, 9.081, 15.783, 18.261, -6.494, 15.228, 20.265, -9.497, 13.5, 21.023, 6.412, 5.165, 11.794, 3.891, -1.759, -3.581, 10.764, -5.481, 21.642, -0.634, 0.392, 20.73, 19.766, 18.544, 8.34, 3.479, 13.276, -5.638, 16.531, 20.26, -5.899, 5.964, 11.724, 13.658, 5.25, 15.671, 11.672, 7.531, 6.758, 25.762, -2.624, 16.966, -14.638, 5.321, 0.165, 13.88, 20.526, 0.087, -11.0, 9.428, 2.711, 14.973, -5.146, 13.944, 5.316, 9.24, 3.606, 17.029, -0.587, 6.523, -2.418, -1.63, 5.506, -0.416, 12.909, 17.032, -0.96, 1.172, 8.671, 15.788, -11.858, 10.639, -5.476]


                state = [0,0,0,0,0,0,0,0,0,0,0,0,0]
                for c in trial['trialdata']['queries']['click']['state']['target']:
                    state[int(c)] = int(trial['trialdata']['stateRewards'][int(c)])
                rews_exp.append(getExpectedReward(state) - len(trial['trialdata']['queries']['click']['state']['target']))

                rew = np.round( trial['trialdata']['score'], 3)
                rews.append(rew)
                rews_norm.append(np.round( rew - opt_scores[trial['trialdata']['trial_id']] ,3))
                rews_good.append(np.round( rew - good_scores[trial['trialdata']['trial_id']] ,3))

                #flowchart
                clicks.append(trial['trialdata']['queries']['click']['state']['target'])
                preselect.append(trial['trialdata']['queries']['preselect']['state']['target'])
                flow_index.append(trial['trialdata']['queries']['flowchart']['index']['target'])
                flow_answer.append(trial['trialdata']['queries']['flowchart']['answer']['target'])
                flow_correct.append(trial['trialdata']['queries']['flowchart']['correct']['target'])

                trial_flow = []
                for presel, fl_ind, fl_ans, fl_cor in zip(preselect[-1], flow_index[-1], flow_answer[-1], flow_correct[-1]):
                    obj = {}
                    obj['preselect'] = presel
                    obj['flow_index'] = fl_ind
                    obj['flow_answer'] = fl_ans
                    obj['flow_correct'] = fl_cor
                    trial_flow.append(obj)

                    flow_index_count += len(fl_ind)
                    flow_correct_count += sum(fl_cor)

                    for fl_ind2, fl_cor2 in zip(fl_ind, fl_cor):
                        trial_flow_all_test[fl_ind2-1] += 1
                        trial_flow_cor_test[fl_ind2-1] += 1 if fl_cor2 else 0

                flowchart.append(trial_flow)

                # bonus computation:
                if condition == 0:
                    for t_c in trial['trialdata']['queries']['click']['state']['time']:
                        i = -1
                        for t_pc in trial['trialdata']['queries']['preselect']['state']['time']:
                            if t_pc is None or t_c < t_pc:
                                break
                            i += 1
                        if trial['trialdata']['queries']['preselect']['clickable']['target'][i]:
                            num_correct_clicks += 1
                        else:
                            num_incorrect_clicks += 1

        # loop thorugh all trials again for practice trials
        count_practice = 0
        for trial in datastring['data']:
            trial_key = reduceNodeString(trial['trialdata']['internal_node_id'])

            if trial_key in ['02100']:
                count_practice += 1
                if(count_practice > min(attempts_practice,10)-3 and count_practice < 11):
                    for fl_ind2, fl_cor2 in zip(trial['trialdata']['queries']['flowchart']['index']['target'], trial['trialdata']['queries']['flowchart']['correct']['target']):

                        for fl_ind, fl_cor in zip(fl_ind2, fl_cor2):
                            trial_flow_all_train[fl_ind-1] += 1
                            trial_flow_cor_train[fl_ind-1] += 1 if fl_cor else 0


    if count_ML_trials < 10:
        print("Aborted Experiment: {} testing trials".format(count_ML_trials))
        cheat_trials = 10-count_ML_trials
    trial_time_mean = np.mean(trial_time)

    flow_correct_perc = 0 if flow_index_count==0 else flow_correct_count/flow_index_count
    flow_correct_perc_training = np.around(np.divide(trial_flow_cor_train, trial_flow_all_train), decimals=2)
    flow_correct_perc_testing = np.around( np.divide(trial_flow_cor_test,  trial_flow_all_test)  , decimals=2)


    return {'WorkerId' : worker_id,
           'hitId': hit_id,
            'assignmentId': assignment_id,
            'condition': condition,
            'condition_type': condition_type,
           'status': status,
           'beginHit': begin_hit,
           'beginExp': begin_exp,
           'endHit': end_hit,
           'totalTime': total_time,
           'attempts_quiz': attempts_quiz,
           'attempts_quiz2': attempts_quiz2,
           'attempts_practice': attempts_practice,
           'genEdu': genEdu ,
           'comEdu': comEdu,
           'age': age,
           'gender': gender,
            'qUnd': qUnd,
           'qConf': qConf,
           'qNotUnd': qNotUnd,
           'qYouAsk': qYouAsk,
           'feedback': feedback,
           'cheatTrials': cheat_trials,
           'trialTimes': trial_time,
           'trialTimeMean': trial_time_mean,
           'timesTimesFinish': trial_time_finish,
           'numberClicks': number_clicks,
           'numberCorClicks': num_correct_clicks,
           'numberIncClicks': num_incorrect_clicks,
           'clicks': clicks,
           'preselect': preselect,
           'flowIndex': flow_index,
           'flowAnswer': flow_answer,
           'flowCorrect': flow_correct,
           'flowIndexCount':  flow_index_count,
           'flowCorrectPerc': flow_correct_perc,
            'flowIndexTraining': trial_flow_all_train,
            'flowCorrectPercTraining': flow_correct_perc_training,
            'flowIndexTesting': trial_flow_all_test,
            'flowCorrectPercTesting': flow_correct_perc_testing,
           'flowchart': flowchart,
           'time_tracking': time_tracking,
           'testingTrials': ML_test_trials,
           'addTrials': ML_add_trials,
           'datastring': datastring,
           'rews': rews,
           'total_rew': np.sum(rews),
           'mean_rew': np.round(np.mean(rews), 2),
           'rews_norm': rews_norm,
           'total_rew_norm': np.sum(rews_norm),
           'mean_rew_norm': np.round(np.mean(rews_norm),2),
           'rews_good': rews_good,
          'total_rew_good': np.sum(rews_good),
          'mean_rew_good': np.round(np.mean(rews_good),2),
          'rews_exp': rews_exp,
          'total_rew_exp': np.sum(rews_exp),
          'mean_rew_exp': np.round(np.mean(rews_exp),2),
          'scores': scores}


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
