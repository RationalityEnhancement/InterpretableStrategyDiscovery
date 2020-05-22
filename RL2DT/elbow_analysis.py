import numpy as np
import os
import gc
import argparse
import pickle

from demonstration_sampling import SamplingMachine, get_state_action_representation, join
from strategy_demonstrations import reward
from mouselab_utils import load_demos

def generate_DSL_file(pos_input, neg_input, info):
    '''
    Generate a pickle file with the matrix that encodes the predicate
    valuations for the input demos.

    Parameters
    ----------
    pos_input: [ (Trial, action) ]
            Demonstrated (state, action) pairs
    neg_input: [ (Trial, action) ]
            (state, action) pairs for encountered states and non-optimal actions
            according to the RL policy

    Returns
    -------
    dsl_data : str
        Path to the binary DSL vectors file.
    '''
    ## calling SM with empty path_to_DSL_data argument creates a pkl file
    ## with a DSL version of the input data
    SamplingMachine(pos_input, neg_input, use_DSL_features=True, 
                    path_to_DSL_data='', info=info)
        
    dsl_data = os.getcwd() + '/PLP_data/DSL_' + info + '.pkl'
    
    return dsl_data

def get_clustering_values(machine, cluster_nums, info):
    '''
    Cluster the binary DSL vectors to each of the candidate number
    of clusters and measure the Bayesian heuristic value of each
    to sum it to the clustering value of a candidate number.

    Parameters
    ----------
    machine : SamplingMachine
        The object that does clustering
    cluster_nums : [ int ]
        List of candidate numbers for the numbers of clusters
    info : str
        String determining the name of the file showing the dendrogram

    Returns
    -------
    clustering_values : [ float ]
        List of clustering values for different candidate numbers
    '''
    clustering_values = []
    for num_cl in cluster_nums:
        print("\nCLUSTERING TO {} CLUSTERS\n".format(num_cl))
        machine.cluster(method="hierarchical", vis=False, info=info,
                        num_clusters=num_cl)
    
        clustering = machine.interpret_clusters()
        score = machine.evaluate_clusters(clustering)
        clustering_values.append(score)
        
    return clustering_values

def save_clustering_values_to_file(data, cluster_sizes, info=''):
    '''
    Save a list of clustering values to csv file in interprets/elbow folder.

    Firs row: candidate numbers
    Second row: clustering values

    Parameters
    ----------
    data : [ float ]
        List with clustering values
    cluster_sizes : [ int ]
        List of candidate numbers for the numbers of clusters
    '''
    cwd = os.getcwd()
    folder_path = cwd+'/interprets/elbow/'
    import csv
    clustering_values_file = 'elbow_'+info+'.csv'
    with open(folder_path + clustering_values_file, 'w') as handle:
        writer = csv.writer(handle)
        writer.writerows([cluster_sizes, data])

def get_elbows(demo_path, num_demos, num_candidates, environment='standard',
               candidate_clusters=list(range(1,26)) + [30,40,50,75,100,200,300], 
               dsl_data=None, info=''):
    '''
    Select the most promising candidates for the numers of clusters in the DSL data

    Parameters
    ----------
    demo_path : str
        Path to the demonstrations
    num_demos : int 
        How many demos to use for interpretation
    num_candidates : int
        How many candidates for the numbers of clusters to select
    environment (optional) : str
        Name of the environment chosen to create the files with demonstrations
        'standard' does not use any name and searches for the files using only
        BASE_NUM_DEMOS parameter
    candidate_clusters (optional) : [ int ] 
        Which numers for the numbers of clusters to test
    dsl_data (optional) : str
        Path to the file with DSL vectors or None, if it is to be generated
    info (optional) : str
        What information to add when naming the file with clustering values
        using the template 'elbow_<num_data>_<info>'

    Returns
    -------
    candidate_elbows : [ int ]
        List of the most promising numbers for the numbers of clusters
    '''

    demos = load_demos(environment, num_demos, except_path=demo_path)

    counter = 0
    for d in demos['pos']:
        s, a = get_state_action_representation(d)
        print("{}: {}: {}".format(counter,s,a))
        counter += 1

    if info == '':
        info = str(num_demos) + '_' + environment if environment != 'standard' \
               else str(num_demos)

    if dsl_data == None:
        path_dsl_data = generate_DSL_file(demos['pos'], demos['neg'], info)
    else:
        path_dsl_data = os.getcwd() + '/PLP_data/' + dsl_data + '.pkl'
    
    sm = SamplingMachine(demos['pos'], demos['neg'], use_DSL_features=True, 
                         path_to_DSL_data=path_dsl_data, all_data=demos)
    clustering_values = get_clustering_values(sm, candidate_clusters, info)
    save_clustering_values_to_file(clustering_values, candidate_clusters, info)

    stacked_clustering_values = zip(clustering_values, clustering_values[1:])
    increases = [next - prev for prev, next in stacked_clustering_values]
    candidates = zip(candidate_clusters[1:], increases)
    zipped_candidate_elbows = sorted(candidates, 
                                     key=lambda x: x[1], 
                                     reverse=True)[:num_candidates]
    candidate_elbows = list(zip(*zipped_candidate_elbows))[0]
    
    print("\nTHE MOST PROMISING CLUSTER NUMBERS ARE: {}".format(candidate_elbows))
    
    return candidate_elbows, path_dsl_data


if __name__ == "__main__":
 
    cwd = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_path', '-d', type=str, help='', 
                        default='')
    parser.add_argument('--num_demos', '-n', type=int, 
                        help='How many demos to use/were used in the DSL data generation.',
                        default=64)
    parser.add_argument('--info', '-i', type=str, 
                        help='What to add to the name of the elbow file.',
                        default='')
    parser.add_argument('--name_dsl_data', '-dsl', type=str, 
                        help='Name of the DSL data file if it was already generated.',
                        default='')
    parser.add_argument('--num_candidates', '-nc',
                        type=int,
                        help="The number of candidates for the number of " \
                            +"clusters to consider.",
                        default=4)
    parser.add_argument('--environment', '-e', type=str, default='standard')
    parser.add_argument('--candidate_clusters', '-c', nargs='+', type=int, 
                        help='What candidate numbers to check.',
                        default=list(range(10)))

    args = parser.parse_args()

    print('                 Getting {} requested elbows\n\n'.format(args.num_candidates))

    get_elbows(args.demo_path, args.num_demos, args.num_candidates, args.environment, 
               args.candidate_clusters, args.name_dsl_data, args.info)
