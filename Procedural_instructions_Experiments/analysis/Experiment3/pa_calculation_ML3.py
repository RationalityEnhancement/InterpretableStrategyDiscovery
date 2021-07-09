import numpy as np



def get_agreement_trial(actions):
    print('')
    print(' Neweeew Trial')

    good_clicks = 0
    bad_clicks = 0
    counter = 0
    k = 6

    for action in actions:

        if action == 0 or (k > 0 and counter+1 > k):
            break

        counter+= 1

        if action in [3,4,7,8,11,12]:
            good_clicks += 1
        else:
            bad_clicks += 1


    sequencewise_agreement = 1 if (bad_clicks == 0) else 0
    click_agreement_ratio = float(good_clicks)/max(1,counter)

    print('')
    #print('good: {}, count: {}'.format(good_clicks, counter))
    print(actions)
    print('PA: {}'.format(click_agreement_ratio))

    return click_agreement_ratio, sequencewise_agreement, 0


def get_agreement_participant(participant_data):
    click_agreement_ratios = []
    sequencewise_agreements = []
    mean_run_lengths = []

    #participant_state_actions = get_state_actions(participant_data)
    #for sequence in participant_state_actions:

    for trial in participant_data:
        list_of_strings = trial['queries']['click']['state']['target']
        list_of_actions = [int(s) for s in list_of_strings] + [0]

        click_agreement_ratio, sequencewise_agreement, mean_run_length  = get_agreement_trial(list_of_actions)
        click_agreement_ratios.append(click_agreement_ratio)
        sequencewise_agreements.append(sequencewise_agreement)
        mean_run_lengths.append(mean_run_length)

    return click_agreement_ratios, mean_run_lengths

def get_agreement_sample(data):
    click_agreement_ratios_sample = []
    click_agreement_means_sample = []

    mean_run_lengths_sample = []
    counter = 0
    print(len(data))
    for participant in data:
        #print(counter)
        counter += 1

        click_agreement_ratios, mean_run_lengths  = get_agreement_participant(participant)

        click_agreement_ratios_sample.append(click_agreement_ratios)
        click_agreement_means_sample.append(np.round(np.mean(click_agreement_ratios),4))

        mean_run_lengths_sample.append(mean_run_lengths)


    return {'click_agreement_ratios_sample': click_agreement_ratios_sample,
            'click_agreement_means_sample': click_agreement_means_sample,
            'mean_run_lengths_sample': mean_run_lengths_sample}
