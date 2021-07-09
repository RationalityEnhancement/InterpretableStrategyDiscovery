import numpy as np
import random

# PAIR AGREEMENT ROAD TRIP

def get_state_actions(trial_info):

    # initial
    states = []
    actions = []
    clicks = trial_info['clicks'].copy()
    clicks.append(-1)
    state = np.zeros(len(trial_info['nodes']))

    for click in clicks:

        # store
        states.append(list(state))
        actions.append(click)

        # recompute state
        if(click > -1):
            state[click] = trial_info['nodes'][str(click)]['reward']

    return states, actions


def correct_clicks(state, trial_info, method = 'graph-distance'):

    # inital
    correct_clicks = []
    visited_nodes = []
    next_nodes = trial_info['end_nodes'].copy()

    if method == 'graph-distance':
        # breadth first traversal starting from terminal nodes
        # until unclicked level reached or all nodes uncovered
        while(len(next_nodes) > 0 and len(correct_clicks) == 0):

            # update
            current_nodes = list(dict.fromkeys(next_nodes))
            visited_nodes += current_nodes
            next_nodes = []

            for node in current_nodes:

                 # check if unobserved
                if state[node] == 0:
                    correct_clicks.append(node)

                # unlock next level
                for next_node in trial_info['nodes'][str(node)]['outEdge']:
                    if next_node not in visited_nodes:
                        next_nodes.append(next_node)


    elif method == 'spatial-distance':
        # traverse from most far away to closest in batches of 2

        nodes_ordered = dict(sorted(trial_info['nodes'].items(), key=lambda item: item[1]['x']))
        nodes_ordered = dict(filter(lambda item: state[item[1]['id']] == 0, nodes_ordered.items()))
        for node in list(nodes_ordered.items())[-2:]:
            correct_clicks.append(int(node[1]['id']))

    else:
        print('Error no method specified')
    return correct_clicks


def get_agreement_trial(trial_info, k, pa_measure, et_mehod):

    # initial
    good_clicks = 0
    bad_clicks = 0
    counter = 0
    states, actions = get_state_actions(trial_info)

    print('')
    print('New trial ---')

    if k == 'end_nodes':
        gt = []
        for node in trial_info['nodes']:
            gt.append(trial_info['nodes'][node]['reward'])
        num = sum([1 if g >= 100 else 0 for g in gt])
        k = num

    # loop through sequence
    for state, action in zip(states, actions):

        # analyis first k clicks
        print('')
        print('State: {}'.format(state))

        if action == -1 or (k > 0 and counter+1 > k):
            break

        counter+= 1
        print('Action: {}'.format(action))


        if action in correct_clicks(state, trial_info, pa_measure):
            good_clicks += 1
            print('good click')

        else:
            bad_clicks += 1
            print('bad click')


    # evaluation
    sequencewise_agreement = 1 if (bad_clicks == 0) else 0
    #click_agreement_ratio = np.round(float(good_clicks) / (good_clicks + bad_clicks), 3)
    click_agreement_ratio = float(good_clicks)/counter

    print('PA: {}'.format(click_agreement_ratio))
    return click_agreement_ratio, sequencewise_agreement, 0


def get_agreement_participant(participant_data, k, n, pa_measure, et_mehod):
    click_agreement_ratios = []
    sequencewise_agreements = []
    mean_run_lengths = []
    counter = 0

    participant_data = eval(participant_data)
    for trial_info in participant_data:

        counter+= 1
        if n > 0 and counter > n:
            break

        click_agreement_ratio, sequencewise_agreement, mean_run_length  = get_agreement_trial(trial_info, k, pa_measure, et_mehod)
        click_agreement_ratios.append(click_agreement_ratio)
        sequencewise_agreements.append(sequencewise_agreement)
        mean_run_lengths.append(mean_run_length)

    return click_agreement_ratios, mean_run_lengths

def get_agreement_sample(data, k=-1, n=-1, pa_measure = 'graph-distance', et_mehod = 'expectation'):
    click_agreement_ratios_sample = []
    click_agreement_means_sample = []
    mean_run_lengths_sample = []

    for participant_data in data:
        print('\n New Participant ------- ')

        click_agreement_ratios, mean_run_lengths  = get_agreement_participant(participant_data, k, n, pa_measure, et_mehod)

        click_agreement_ratios_sample.append(click_agreement_ratios)
        click_agreement_means_sample.append(np.round(np.mean(click_agreement_ratios),4))
        mean_run_lengths_sample.append(mean_run_lengths)


    return {'click_agreement_ratios_sample': click_agreement_ratios_sample,
            'click_agreement_means_sample': click_agreement_means_sample,
            'mean_run_lengths_sample': mean_run_lengths_sample}
