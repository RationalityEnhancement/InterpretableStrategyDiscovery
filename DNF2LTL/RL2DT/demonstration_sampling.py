import torch, torch.nn as nn, torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as distances
import random
import os
import gc
import argparse
import pickle

from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.sparse import vstack
from sklearn.manifold import TSNE
from itertools import cycle
from RL2DT.decision_tree_imitation_learning import run_all_programs_on_demonstrations, train
from RL2DT.hyperparams import MAX_VAL, NUM_PROGRAMS, NUM_ACTIONS, NUM_CLUSTERS, ENV_TYPE, \
                        CLUSTER_DEPTH, REJECT_VAL
from RL2DT.strategy_demonstrations import reward
from RL2DT.mouselab_utils import join, join_distrs

from RL2DT.PLP.DSL import *

DEBUG = False ## use to print out the data in the clusters

def wrapper_train(tree_depth, demos, validation_demos, pred_data=[None,None], verbose=True):
    """
    Wrapper function for the 'train' function. 

    See 'decision_tree_imitation_learning.train'.
    """
    return train(program_gen_step_size = 1000, 
                 num_programs = NUM_PROGRAMS, 
                 num_dts = 5, 
                 max_num_particles = 25, 
                 input_demos = demos, 
                 further_demos = validation_demos, 
                 tree_depth = tree_depth, 
                 return_prior=True,
                 pred_data=pred_data,
                 verbose=verbose)

def taxi_distance(x,y):
    """
    Wrapper function that defines the \ell_1 norm.
    """
    return distances.minkowski(x,y,p=1)

def cluster_hierarchical(vectors, num_cl=None, vis=False, info=''):
    """
    Run the hierarchical clustering algorithm. 


    Parameters
    ----------
    vectors : ndarray
        vectors.shape = (n_samples, n_features)
    num_cl (optional) : int
        The number of clusters to form using the hierarchical method
    vis (optional) : bool
        Used to output the found dendrogram

    Returns
    -------
    labels : [ int ]
        Cluster classification for input 'vectors'
    num_cl : int
    """
    linked = linkage(vectors, method='average', metric=taxi_distance)
    labels = range(vectors.shape[0])
    if vis: 
        plt.figure(figsize=(40, 28))
    dendrogram(linked,
               orientation='top',
               labels=labels,
               distance_sort='descending',
               show_leaf_counts=True)
    if vis: 
        plt.savefig('./dendrograms/dendro_'+info+'.png')
    
    if num_cl is not None:
        labels = fcluster(linked, num_cl, criterion='maxclust')
        return labels, num_cl 


def get_state_action_representation_from_DSL(demos):
    """
    Convert demonstrations into a matrix of predicate values.

    See 'decision_trer_imitation_learning.run_all_programs_on_demonstrations'.

    Parameters
    ----------
    demos : dict

    Returns
    -------
    ndarray
        Dense representation of the csr_matrix
    X_csr : csr_matrix
    y : [ int(bool) ]
    """
    input_demos = demos
    if type(demos) is list:
        input_demos = {'pos' : demos, 'neg' : []}
    X_csr, y = run_all_programs_on_demonstrations(NUM_PROGRAMS, input_demos)
    X = X_csr.todense()
    return np.array(X), X_csr, y

def get_state_action_representation(demo):
    """
    Convert demonstrations into a representation acceptable by a neural network.

    Parameters
    ----------
    demo : ( Trial, int )

    Returns
    -------
    output_state = [ int ]
        A vector of observed values; non-observed values are 0's
    [ int ]
        One-element list with the action taken in the demo
    """
    state, action = demo
    observed = {n.label: n.value for n in state.observed_nodes if n.label != 0}
    len_state = len(state) if type(state) is list else state.num_nodes
    output_state = [0] * len_state
    for node, value in observed.items():
        output_state[node] = value

    return output_state, [action]

def cluster_mean_shift(vectors):
    """
    Cluster input vectors using the mean shift algorithm (Comaniciu & Meer, 2002).

    See https://scikit-learn.org/stable/modules/clustering.html.

    Parameters
    ----------
    vectors : ndarray
        vectors.shape = (n_samples, n_features)

    Returns
    -------
    labels : [ int ]
        Cluster classification for input 'vectors'
    num_cl : int
        The number of discovered clusters
    """
    ms = MeanShift()
    ms.fit(vectors)

    labels = ms.labels_
    labels_unique = np.unique(labels)
    n_clusters = len(labels_unique)

    print("Discovered {} clusters".format(n_clusters))
    print(labels)

    return labels, n_clusters


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

def init_weights(m):
    """
    Initialize the weights of a neural network.
    """
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.1, std=0.01)
        nn.init.constant_(m.bias, 0.0)

class Decoder(nn.Module):
    """
    The decoder network for states outputted by get_state_action_representation. 

    Can decode the state or predict the action.
    """

    def __init__(self, cont_output_size, cat_output_size, input_size, 
                 hidden_size):
        
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.cont_output_size = cont_output_size
        self.cat_output_size = cat_output_size
        self.input_size = input_size

        self.decoder_old = nn.Sequential(
            nn.Linear(input_size, int(hidden_size/4)),
            nn.ReLU(),
            nn.BatchNorm1d(int(hidden_size/4)),
            nn.Linear(int(hidden_size/4),int(hidden_size/2)),
            nn.ReLU(),
            nn.BatchNorm1d(int(hidden_size/2)),
            nn.Linear(int(hidden_size/2),hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size))

        self.action_decoder_old = nn.Sequential(
            nn.Linear(hidden_size, cat_output_size),
            nn.Softmax())

        self.action_decoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, cat_output_size))

        self.state_decoder = nn.Linear(hidden_size, cont_output_size)

        self.apply(init_weights)

    def forward(self, hidden_rep):
        action = self.action_decoder(hidden_rep)
        print('Predic: {}'.format([list(a).index(max(a)) for a in action]))
        return None, action


class Encoder(nn.Module):
    """
    The encoder network for states outputted by get_state_action_representation. 

    Encodes the input into a 4 times smaller continuous representation.
    """

    def __init__(self, state_input_size, action_input_size, hidden_size, 
                 hidden_rep_size):
        
        super(Encoder, self).__init__()

        self.hidden_rep_size = hidden_rep_size
        self.state_input_size = state_input_size
        self.action_input_size = state_input_size

        self.encoder = nn.Sequential(
            nn.Linear(state_input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, int(hidden_size/2)),
            nn.ReLU(),
            nn.BatchNorm1d(int(hidden_size/2)),
            nn.Linear(int(hidden_size/2), int(hidden_size/4)),
            nn.ReLU(),
            nn.BatchNorm1d(int(hidden_size/4)),
            nn.Linear(int(hidden_size/4),hidden_rep_size))

        self.apply(init_weights)

    def forward(self, s, a):
        hidden_rep  = self.encoder(s)
        return hidden_rep

class Autoencoder(nn.Module):
    """
    The 'Encoder' plus the 'Decoder' networks. 
    """

    def __init__(self, 
                 state_input_size, 
                 action_input_size, 
                 num_actions, 
                 hidden_size, 
                 hidden_rep_size):
        
        super(Autoencoder, self).__init__()

        self.encoder = Encoder(state_input_size, action_input_size,
                               hidden_size, hidden_rep_size)
        self.decoder = Decoder(state_input_size,num_actions, 
                               hidden_rep_size, hidden_size)

    def forward(self, s, a):
        encoded                       = self.encoder(s,a)
        decoded_state, decoded_action = self.decoder(encoded)
        return encoded, decoded_state, decoded_action


class DecoderDSL(nn.Module):
    """
    The decoder network for states outputted by get_state_action_representationDSL.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(DecoderDSL, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.decoder = nn.Sequential(
            nn.Linear(input_size, int(hidden_size/4)),
            nn.ReLU(),
            nn.BatchNorm1d(int(hidden_size/4)),
            nn.Linear(int(hidden_size/4),int(hidden_size/2)),
            nn.ReLU(),
            nn.BatchNorm1d(int(hidden_size/2)),
            nn.Linear(int(hidden_size/2),hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, output_size))

        self.apply(init_weights)

    def forward(self, hidden_rep):
        decoded = self.decoder(hidden_rep)
        return decoded


class EncoderDSL(nn.Module):
    """
    The encoder network for states outputted by get_state_action_representationDSL. 

    Encodes the input into a 4 times smaller continuous representation.
    """

    def __init__(self, input_size, hidden_size, hidden_rep_size):
        super(EncoderDSL, self).__init__()

        self.hidden_rep_size = hidden_rep_size
        self.input_size = input_size

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, int(hidden_size/2)),
            nn.ReLU(),
            nn.BatchNorm1d(int(hidden_size/2)),
            nn.Linear(int(hidden_size/2), int(hidden_size/4)),
            nn.ReLU(),
            nn.BatchNorm1d(int(hidden_size/4)),
            nn.Linear(int(hidden_size/4),hidden_rep_size))

        self.apply(init_weights)

    def forward(self, s):
        hidden_rep  = self.encoder(s)
        return hidden_rep

class AutoencoderDSL(nn.Module):
    """
    The EncoderDSL plus the DecoderDSL networks. 
    """

    def __init__(self, input_size, hidden_size, hidden_rep_size):
        super(AutoencoderDSL, self).__init__()

        self.encoder = EncoderDSL(input_size, hidden_size, hidden_rep_size)
        self.decoder = DecoderDSL(hidden_rep_size, hidden_size, input_size)

    def forward(self, s):
        encoded = self.encoder(s)
        decoded = self.decoder(encoded)
        return encoded, decoded


def print_clusters(vectors, labels, nclusters, show=False):
    """
    Graphically show the clustering of the hidden representations.

    Saves the plot in 'cwd/plots' directory.

    Parameters
    ----------
    vectors : ndarray
        vectors.shape = (n_samples, n_features)
    labels : [ int ]
        The cluster classification for input 'vectors'
    nclusters : int
        The number of clusters
    show (optional) : bool
        Outputs the plot to the terminal
    """
    plt.figure(1)
    plt.clf()

    vecs2D = TSNE(n_components=2).fit_transform(vectors)

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(nclusters), colors):
        my_members = labels == k

        cluster_vecs2D = vecs2D[my_members, :]

        print(cluster_vecs2D)
        print(cluster_vecs2D[:,0])
        print(cluster_vecs2D[:,1])

        plt.scatter(cluster_vecs2D[:,0], 
                    cluster_vecs2D[:,1], 
                    c=col, 
                    label='cluster {}'.format(k))

    plt.title('Estimated clusters')
    plt.legend()

    if show:
        plt.show()

    cwd = os.getcwd()
    if not os.path.exists(cwd+"/plots"):
        os.makedirs(cwd+"/plots")
    plt.savefig(cwd+'/plots/clusters.png')


class SamplingMachine(object):
    """
    A class for that enables sampling (state, action) pairs from the demonstrations.

    It computes an embedding of input demos or uses them in their original form.

    It clusters the data with one of the implemented clustering algorithms: 
    mean shift, hierarchical clustering or a custom callable method.

    It allows sampling a proportion of elements from all or some of the clusters.

    Mainly appropriate for discrete environments.
    """

    def __init__(self, demonstrations, negative_demonstrations, lr=0.0001, 
                 hidden_size=128, hidden_rep_size=8, num_actions=13, 
                 use_DSL_features=False, path_to_DSL_data=None, info='',
                 all_data=None):
        
        self.demos = demonstrations
        self.neg_demos = negative_demonstrations
        self.hidden_size = hidden_size
        self.hidden_rep_size = hidden_rep_size
        self.lr = lr
        self.all_data = all_data

        self.one_hots = np.eye(num_actions)
        self.input_states = []
        self.input_actions = []
        
        def normalize_and_stack(states):
            return (np.stack(states) + MAX_VAL) / (2 * MAX_VAL)

        if not use_DSL_features:
            for demo in demonstrations:
                s, a = get_state_action_representation(demo)
                self.input_states.append(s)
                # Uncomment to turn the action into a one-hot representation
                #a = self.one_hots[a[0]]
                self.input_actions.append(a)
                print("{}: {}".format(s,a))
            self.input_states = normalize_and_stack(self.input_states)
            self.input_actions = np.stack(self.input_actions)
            self.pipeline_X = None
            self.pipeline_y = None

        else:
            print("  Getting state-action representation from DSL...")
            if path_to_DSL_data:
                with open(path_to_DSL_data, 'rb') as handle:
                    data_matrix, X, y = pickle.load(handle)
                    print(X.shape)
            else:
                dct = {'pos':demonstrations, 'neg':negative_demonstrations}
                data_matrix, X, y = get_state_action_representation_from_DSL(dct)
                self.save_data((data_matrix, X, y), info)
            self.input_states = data_matrix[:len(demonstrations), :]
            self.pipeline_X = X
            self.pipeline_y = y
            print("  Done")

        self.init_pipeline_X = self.pipeline_X
        self.init_pipeline_y = self.pipeline_y

        self.num_inputs = len(self.input_states)
        self.num_actions = num_actions

        self.state_size = len(self.input_states[0])
        if self.input_actions != []:
            self.action_size = len(self.input_actions[0]) 
        else:
            self.action_size = None

        if use_DSL_features:
            self.TrajectoryAutoencoder = AutoencoderDSL(self.state_size, 
                                                        self.hidden_size, 
                                                        self.hidden_rep_size).to(device)
        else:
            self.TrajectoryAutoencoder = Autoencoder(self.state_size,
                                                     self.action_size, 
                                                     self.num_actions, 
                                                     self.hidden_size, 
                                                     self.hidden_rep_size).to(device)

        self.hidden_reps = None
        self.labels = []
        self.n_clusters = 0
        self.clustering_methods = {'mean_shift': cluster_mean_shift, 
                                   'hierarchical': cluster_hierarchical}
        self.clustering_value = None

        def mse_crossentropy_loss(pred_state, target_state, pred_act, target_act):
            state_sum_diff = ((pred_state - target_state) ** 2).sum()
            state_n_elems = pred_state.data.nelement()
            cross_entropy = nn.CrossEntropyLoss()(pred_act, target_act)
            return  state_sum_diff / state_n_elems + cross_entropy

        def mse_loss(pred_state, target_state, pred_act, target_act):
            state_sum_diff = ((pred_state - target_state) ** 2).sum()
            state_n_elems = pred_state.data.nelement()
            act_sum_diff = ((pred_act - target_act) ** 2).sum()
            act_n_elems = pred_state.data.nelement()
            return state_sum_diff / state_n_elems + act_sum_diff / act_n_elems

        def act_mse_loss(pred_state, target_state, pred_act, target_act):
            act_sum_diff = ((pred_act - target_act) ** 2).sum()
            act_n_elems = pred_state.data.nelement()
            return act_sum_diff / act_n_elems
 
        def act_crossentropy_loss(pred_state, target_state, pred_act, target_act):
            return nn.CrossEntropyLoss()(pred_act, target_act)

        def mse_dsl_loss(pred, target):
            return ((pred - target) ** 2).sum() / pred.data.nelement()

        self.criterion = mse_dsl_loss
        self.optimizer = optim.Adam(self.TrajectoryAutoencoder.parameters(), 
                                    lr=self.lr)
        
    def save_data(self, tuple_to_save, add_info):
        """
        Save some data to a pickle.
        
        Parameters
        ----------
        tuple_to_save : tuple
        add_info : str
            An additional string to add to the name
        """
        cwd = os.getcwd()
        folder_path = cwd+'/PLP_data/'
        if add_info != '':
            add_info = '_' + add_info
        
        newpath = folder_path + 'DSL' + add_info + '.pkl'
        try:
            with open(newpath, 'wb') as handle:
                pickle.dump(tuple_to_save, handle)
        except:
            return 0
        return 1

    def train(self, epochs, batch_size, num_demos):
        """
        Parameters
        ----------
        epochs : int
        batch_size : int
        num_demos : int
            The number of demonstrations the clicks for clustering come from
        """
        cwd = os.getcwd()
        folder_path = cwd+'/autoencoder_runs/'
        f = str(epochs)+'_batch_size'+str(batch_size)+'_demo_size'+str(num_demos)
        newpath = folder_path + f
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        train_losses = []

        for epoch in range(epochs):
            permutation = np.random.permutation(self.num_inputs)
            total_loss = 0

            for i in range(0, self.num_inputs, batch_size):
                self.optimizer.zero_grad()

                indices = permutation[i : i+batch_size]
                selected_states = self.input_states[indices]
                batch_state = torch.FloatTensor(selected_states).to(device)
                if self.pipeline_X == None:
                    selected_actions = self.input_actions[indices]
                    batch_action = torch.FloatTensor(selected_actions).to(device)
                    selected_actions = [a[0] for a in selected_actions]
                    target_action = torch.LongTensor(selected_actions).to(device)  

                    res = self.TrajectoryAutoencoder(batch_state, batch_action)
                    batch_decoded_state, batch_decoded_action = res[1], res[2]
                    loss = self.criterion(batch_decoded_state, 
                                          batch_state, 
                                          batch_decoded_action, 
                                          batch_action)
                else:
                    res = self.TrajectoryAutoencoder(batch_state)
                    batch_decoded_state = res[1]
                    loss = self.criterion(batch_decoded_state, batch_state)

                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                torch.cuda.empty_cache()

                del batch_state
                del batch_decoded_state
                del loss

            gc.collect()

            mean_total_loss = total_loss / (self.num_inputs / batch_size)
            train_losses.append(mean_total_loss)
            print ("Epoch {}, Loss: {}".format(epoch, mean_total_loss))

            if (epoch+1) % 10 == 0:
                self.checkpoint(epoch, 
                                train_losses, 
                                path = newpath + '/autoencoder_' + str(epoch))

    def get_hidden_reps(self, trajs=None):
        """
        Compute hidden representations of input vectors.

        Parameters
        ----------
        trajs (optional) : [ ndarray, ndarray ]
            trajs[0].shape = (n_samples, n_features)
            trajs[1].shape = (n_samples, 1)
            A list of data fed to the network.
        """
        print(self.hidden_reps)
        if self.hidden_reps is not None:
            return self.hidden_reps

        input_states = self.input_states if trajs is None else trajs[0]
        input_actions = self.input_actions if trajs is None else trajs[1]
        states = torch.FloatTensor(input_states).to(device)
        actions = torch.FloatTensor(input_actions).to(device)

        self.TrajectoryAutoencoder.eval()
        hidden_reps, _, _ = self.TrajectoryAutoencoder(states, actions)

        self.hidden_reps = hidden_reps.data.cpu().numpy()

        return hidden_reps.data.cpu().numpy()

    def cluster(self, vectors=None, method='mean_shift', **kwargs):
        """
        Perform a clustering.

        Parameters
        ----------
        vectors (optional) : ndarray
            trajs[0].shape = (n_samples, n_features)
        method (optional) : Any
            A string denoting the algorithm to use when clustering or
            a function that performs it

        Returns
        -------
        labels : [ int ]
            Cluster classification for input 'vectors'
        n_clusters : int
            The number of discovered clusters
        """
        assert method in ['mean_shift', 'hierarchical'] or callable(method), \
        "Only MeanShift, Dendrogram or custom clustering methods supported \
        ('mean_shift'; 'hierarchical'; <class 'function'> followed by its \
        arguments with 'nclusters' as one of them)"

        if method == 'mean_shift':
            input_vectors = self.get_hidden_reps() if vectors == None else vectors
            labels, n_clusters = cluster_mean_shift(input_vectors)

        elif method == 'hierarchical':
            input_vectors = self.input_states
            try:
                vis = kwargs['vis']
                add_info = kwargs['info']
            except:
                vis = False
                add_info = ''
            try:
                num_cl = kwargs['num_clusters']
            except:
                num_cl = NUM_CLUSTERS
            labels, n_clusters = cluster_hierarchical(vectors=input_vectors, 
                                                      num_cl=num_cl, 
                                                      vis=vis,
                                                      info=add_info)

        elif callable(method):
            for key, val in kwargs.items():
                if key == 'n_clusters':
                    n_clusters = val
            labels = method(**kwargs)

        self.labels = labels
        self.n_clusters = n_clusters

        return labels, n_clusters

    def get_ordered_clusters(self, labels=None, num_clusters=None, order='biggest',
                             split=0.7, clust_max_dep=3):
        """
        Order clusters according to some measure and return the first 'num_clusters'.

        Parameters
        ----------
        labels (optional) : [ int ]
            Cluster classification for some set of vectors
        num_clusters (optional) : int
            The number of clusters to return; default is all found
        order (optional) : Any
            A string denoting the measure to use when comparing two clusters;
            'biggest' comapres cluster's sizes, 'random' decides randomly,
            'value' decides based on the value of the cluster (complexity)
        split (optional) : float in [0,1]
            In what proportion to split every cluster when computing its formula
        clust_max_dep (optional) : int
            Maximum depth of a formula computed for each cluster when 'value' 
            ordering is chosen

        Returns
        -------
        clusters : [ int ]
            The first 'num_clusters' labels for sorted clusters
        """
        assert order in ['value', 'biggest', 'random'], \
              "Available ordering: biggest, random, value; not {}".format(order)

        labels = self.labels if labels is None else labels
        num_clusters = self.n_clusters if num_clusters is None else num_clusters

        unique_labels = set(labels)
        cluster_sizes = {u: list(labels).count(u) for u in unique_labels}
        ############ removing small clusters which are unreliable ############
        unique_labels = [l for l in unique_labels \
                         if cluster_sizes[l] > REJECT_VAL * self.num_inputs]
        self.n_clusters = len(unique_labels)
        ######################################################################

        if order == 'biggest':
            ordered_unique_labels = sorted(unique_labels, 
                                           key=lambda x: cluster_sizes[x], 
                                           reverse=True)
        elif order == 'random':
            ordered_unique_labels = sorted(unique_labels, 
                                           key=lambda x: random.random())
        elif order == 'value':
            if self.clustering_value == None:
                print('Valuation not computed. Computing formulas for clusters...')
                cls = self.interpret_clusters(verbose=False, split=split)
                self.evaluate_clusters(cls)
                print('Done.')
            clust_vals = self.clustering_value
            ordered_unique_labels = sorted(unique_labels,
                                           key=lambda x: clust_vals[x],
                                           reverse=True)

        clusters = ordered_unique_labels[:num_clusters]
        return clusters

    def generate_positive_samples(self, num_samples, num_clusters=None, data=None, 
                                  labels=None, order='biggest', validation=True,
                                  which_cluster=None, clust_max_dep=3):
        """
        Sample a number of demonstrated (state, action) pairs according to the 
        clusters.

        Parameters
        ----------
        num_samples : int or float in [0,1]
        num_clusters (optional) : int
            The number of clusters to sample from; default is all found
        data (optional) : [ (Any, Any)  ]
            The data to cluster assumed to be in relation to the computed clusters
        labels (optional) : [ int ]
        order (optional) : str
        validation (optional) : bool
            Whether to sample the validation set from clusters or not
        which_cluster (optional) : int
            Which cluster to sample from condiering the imposed order
        split (optional) : float in [0,1]
        clust_max_dep (optional) : int

        Returns
        -------
        sampled_data : [ (Any, Any) ]
        validation_sampled_data : [ (Any, Any) ]
        """
        clusters = self.get_ordered_clusters(labels, num_clusters, 
                                             order, num_samples, clust_max_dep)
        data = self.demos if data is None else data
        labels = self.labels if labels is None else labels
        number_samples = num_samples
        cluster_num = True if which_cluster != None else False 

        sampled_data = []
        sampled_indices = []
        if validation: validation_sampled_data = []
        counter = 0
        zero_indices = []

        for c in clusters:
            counter += 1
            cluster_data = [data[i] for i in range(len(data)) if labels[i] == c]
            cluster_indices = [i for i in range(len(data)) if labels[i] == c]
            
            number_samples = max(int(num_samples * len(cluster_data)), 1) if \
                                num_samples <= 1 else num_samples
            sample = random.sample(list(enumerate(cluster_data)), 
                                   min(len(cluster_data), number_samples))
            sample_data = [pair[1] for pair in sample]
            sample_cluster_indices = [pair[0] for pair in sample]
            sample_indices = [cluster_indices[i] for i in sample_cluster_indices]
            sample_zeros = [i for i in range(len(sample_data)) if sample_data[i][1] == 0]

            if (cluster_num and which_cluster == counter) or not cluster_num:
                print("Size of the cluster: {}".format(len(cluster_data)))
                print("Size of the sample: {}".format(len(sample_data)))
           
                sampled_data += sample_data
                sampled_indices += sample_indices
                zero_indices += sample_zeros

            if validation:
                sample_val_data = [cluster_data[i] for i in range(len(cluster_data)) 
                                   if i not in sample_cluster_indices]
                if cluster_num and which_cluster == counter or not cluster_num:
                    validation_sampled_data += sample_val_data


        self.pipeline_X = vstack((self.pipeline_X[sampled_indices, :], 
                                  self.pipeline_X[sum(self.pipeline_y):,:]))
        stacked_y = np.hstack((self.pipeline_y[sampled_indices],
                               [0] * sum(self.pipeline_y == 0)))
        self.pipeline_y = np.array(stacked_y, dtype='uint8')

        if validation:
            return sampled_data, validation_sampled_data, zero_indices
        return sampled_data, zero_indices

    def generate_negative_samples(self, data, sampled_data, zeros=[], validation=False):
        """
        Output negative examples (suboptimal (state, action) pairs according to 
        the policy used to generate the demonstrations) associated with the 
        sampled datapoints.

        Parameters
        ----------
        data : dict
            pos: [ (Any, Any) ]
                Demonstrated (state, action) pairs
            neg: [ (Any, Any) ]
                (state, action) pairs for encountered states and non-optimal 
                actions according to the RL policy
            leng_pos: [ int ]
                Lengths of the consecutive demonstrations
            leng_neg: [ int ]
                Lengths of the negative examples for the consecutive 
                demonstrations
        sampled_data : [ (Any, Any)  ]
        zeros (optional) : [ int ]
            Mouselab ad-hoc. List of indices of positive demos that terminate
        validation (optional) : bool
            Whether the generated set is used in the pipeline

        Returns
        -------
        negative_sampled_data : [ (Any, Any)  ]
        validation_sampled_data : [ (Any, Any)  ]
        """
        negative_sampled_data = []
        negative_sampled_indices = []
        for sample in sampled_data:
            i = data['pos'].index(sample) ## index of a particular move in a demo
            all_num = 0
            for which, num in enumerate(data['leng_pos']):
                all_num += num
                if all_num > i:
                    which_demo = which ## index of a demo the move with index i comes from
                    break

            sum_neg_lengths = sum(data['leng_neg'][:which_demo])

            key = sum_neg_lengths-1 
            value = sum_neg_lengths + data['leng_neg'][which_demo]
            demo_negative_data = data['neg'][key : value]
            state, action = sample
            for demo_state, demo_action in demo_negative_data:
                if demo_state == state:
                    negative_sampled_data.extend([(demo_state, demo_action)])
                    demo_index = data['neg'].index((demo_state, demo_action))
                    negative_sampled_indices.append(demo_index)

        if not validation:
            num_pos = sum(self.pipeline_y == 1)
            num_neg = len(negative_sampled_data)
            pos_sample = self.pipeline_X[:num_pos, :]
            neg_sample = self.pipeline_X[num_pos + negative_sampled_indices, :]
            y_vector = [1] * num_pos + [0] * num_neg
            ######################### Mouselab ad-hc #########################
            ########################## Removing 0's ##########################
            non_zero = [self.pipeline_X[i, :] for i in range(num_pos)
                                              if i not in zeros]
            pos_sample = vstack(non_zero) if non_zero != [] else self.pipeline_X[0,:]
            num_pos = pos_sample.shape[0]
            y_vector = [1] * num_pos + [0] * num_neg
            ##################################################################

            self.pipeline_X = vstack((pos_sample, neg_sample))
            self.pipeline_y = np.array(y_vector, dtype='uint8')
            
        return negative_sampled_data

    def sample_from_clusters(self, num_samples, all_data=None, num_clusters=None, data=None,
                             labels=None, order='biggest', pos_validation=True,
                             neg_validation=False, which_cluster=None,
                             clust_max_dep=CLUSTER_DEPTH):
        """
        Generate sampled positive and negative data for imitation learning.

        See 'generate_(positive|negative)_samples'.

        Parameters
        ----------
        all_data : dict
        num_samples : int or float in [0,1]
            Controls the split
        num_clusters (optional) : int
        data (optional) : [ (Any, Any)  ]
        labels (optional) : [ int ]
        order (optional) : str
        neg_validation (optional) : bool
        pos_validation (optional) : bool
        which_cluster (optional) : int
        split (optional) : float in [0,1]
        clust_max_dep (optional) : int

        Returns
        -------
        pos_result[:-1] : [ (Any, Any) ], [ (Any, Any ) ] (optional)
            Sampled positive and (maybe) validation demonstrations
        neg_demos : [ (Any, Any) ]
            Neative demonstrations associated with the sampled data
        neg_val_demos : [ (Any, Any) ]

        """
        all_data = all_data if all_data != None else self.all_data
        pos_result = self.generate_positive_samples(num_samples, 
                                                    num_clusters=num_clusters, 
                                                    data=data, 
                                                    labels=labels, 
                                                    order=order, 
                                                    validation=pos_validation,
                                                    which_cluster=which_cluster,
                                                    clust_max_dep=clust_max_dep)
        neg_demos = self.generate_negative_samples(data=all_data,
                                                   sampled_data=pos_result[0],
                                                   validation=False,
                                                   zeros=pos_result[-1])

        if DEBUG:
            counterrr = 0
            print('POSITIVE\n')
            for d in pos_result[0] + pos_result[1]:
                s, a = get_state_action_representation(d)
                print("{}: {}: {}".format(counterrr,s,a))
                counterrr += 1
            counterrr = 0
            print('\nNEGATIVE\n')
            for d in neg_demos:
                s, a = get_state_action_representation(d)
                print("{}: {}: {}".format(counterrr,s,a,))
                counterrr += 1



        if neg_validation:
            neg_val_demos = self.generate_negative_samples(data=all_data,
                                                           sampled_data=pos_result[1],
                                                           validation=True)
            return pos_result[:-1] + (neg_demos, neg_val_demos)
        return pos_result[:-1] + (neg_demos,)

    def interpret_clusters(self, split=0.7, all_demos=None, num_clusters=None, 
                                  max_depth=CLUSTER_DEPTH, data=None, labels=None, verbose=True):
        """
        Use the PLP method to describe strategies encompassed by (state, action)
        pairs from each of the clusters.

        Parameters
        ----------
        split (optional) : float
            Training-validation split for clusters
        all_demos (optional) : dict
            pos: [ (Any, Any) ]
                Demonstrated (state, action) pairs
            neg: [ (Any, Any) ]
                (state, action) pairs for encountered states and non-optimal 
                actions according to the RL policy
            leng_pos: [ int ]
                Lengths of the consecutive demonstrations
            leng_neg: [ int ]
                Lengths of the negative examples for the consecutive 
                demonstrations
        num_clusters (optional) : int
            The number of clusters we want to create formulas for; a default is 
            all found
        max_depth (optional) : int
            Maximum depth of a formula for any cluster to consider
        data (optional) : [ (Any, Any) ]
            Clustered (state, action) pairs
        labels (optional) : [ int ]
        verbose (optional) : bool

        Returns
        -------
        cluster_formulas : [ ( StateActionProgram, float ) ]
            Programs found for each cluster alongside its log prior
        """
        all_demos = self.all_data if all_demos is None else all_demos
        clusters = self.get_ordered_clusters(labels, num_clusters)
        data = self.demos if data is None else data
        labels = self.labels if labels is None else labels

        cluster_formulas = []
        counter = 0
        sep = "\n                    "
        for c in clusters:
            counter += 1
            res = self.sample_from_clusters(num_samples=split,
                                            all_data=all_demos,
                                            pos_validation=True, 
                                            neg_validation=True,
                                            which_cluster=counter)
            positive_samples, val_positive_samples = res[0], res[1]
            negative_samples, val_negative_samples = res[2], res[3]
            z = 0
            for d in positive_samples:
                if d[1] == 0: z += 1

            cluster_data = {'pos': positive_samples,
                            'neg': negative_samples}
            val_cluster_data = {'pos': val_positive_samples,
                                'neg': val_negative_samples}

            if verbose: print(sep +"Checking formulas " + \
                              "with max depth {}\n".format(max_depth))

            cluster_formula, value_formula = wrapper_train(max_depth,
                                                           cluster_data, 
                                                           val_cluster_data,
                                                           verbose=verbose,
                                                           pred_data=[self.pipeline_X,
                                                                      self.pipeline_y])
            if cluster_formula is not None:
                print(cluster_formula)

            cluster_formulas.append((c, cluster_formula, value_formula))
            self.reset_pipeline()

        return cluster_formulas

    def evaluate_clusters(self, cluster_formulas, value='weighted_sum'):
        """
        Numerically evaluate the clustering based on the formulas found by the 
        PLP method.

        Parameters
        ----------
        cluster_formulas : [ (int, StateActionProgram, float ) ]
        value : str
            Whether to measure the 'sum' of the MAP estimate or the 'weighted_sum'
            considering the sizes of the clusters

        Returns
        -------
        float
            The mean log prior of the found formulas
        """
        num_elems = len(self.labels)
        total_val = {}
        num_cl = len(cluster_formulas)
        clustered_points_num = 0
        print("\n\n")
        print("Sufficiently big clusters: {}".format(num_cl))
        for c, formula, val in cluster_formulas:
            c_size = len([l for l in self.labels if l == c])
            clustered_points_num += c_size

            if value == 'weighted_sum':
                total_val[c] = val * c_size / num_elems
            elif value == 'sum':
                total_val[c] = val * 1

        clust_val = sum(total_val.values())
        self.clustering_value = total_val
        print("Value of clustering: {}".format(clust_val))
        return clust_val

    def get_nclust(self):
        return self.n_clusters

    def get_X(self):
        return self.pipeline_X

    def get_y(self):
        return self.pipeline_y

    def reset_pipeline(self):
        """
        Set the variables to their initial values.

        Needed when sampling modifies the variables and we want to sample again.
        """
        self.pipeline_X = self.init_pipeline_X
        self.pipeline_y = self.init_pipeline_y

    def checkpoint(self, epoch, losses, path):
        """
        Save the current autoencoder network.

        Parameters
        ----------
        epoch : int
           The number of epochs the autoencoder was trained for
        losses : [ float ]
           Errors obtained in each of the epochs
        path : string
           A path to save the file to
        """
        dct = {'epoch': epoch, 
                 'losses': losses, 
                 'model_state_dict': self.TrajectoryAutoencoder.state_dict()}
        torch.save(dct, path)

    def load_state_dict(self, arg):
        """
        Load a saved autoencoder network.

        Parameters
        ----------
        arg : string
            A path to the '.pth' file containing one of the autoencoder classes
        """
        self.TrajectoryAutoencoder.load_state_dict(torch.load(arg))

    def save(self, num_demos):
        """
        Pickle the current instantiation of the class.

        Parameters
        ----------
        num_demos : int
           The number of demonstrations the data that was clustered came from
        """
        cwd = os.getcwd()
        folder_path = cwd+'/autoencoder_runs/'
        NUM_DEMOS = str(num_demos)
        
        newpath = folder_path + 'sampling_machine_demo_size' + NUM_DEMOS + '.pkl'
        try:
            with open(newpath, 'wb') as handle:
                pickle.dump(self, handle)
        except:
            return 0
        return 1


if __name__ == "__main__":
 
    cwd = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_path', '-d', type=str, help='', 
                        default=cwd+'/PLP_data/copycat_64.pkl')
    parser.add_argument('--num_demos', '-n', type=int, help='',
                        default=64)
    parser.add_argument('--info', '-i', type=str, default='')
    parser.add_argument('--dsl_data', '-s', type=str, help='',
                        default=cwd+'PLP_data/DSL.pkl')
    parser.add_argument('--environment', '-e', type=str, default='standard')
    parser.add_argument('--num_clusters', '-c', type=int, default=NUM_CLUSTERS)

    args = parser.parse_args()

    num_demos = args.num_demos
    add_info = args.info

    try:
        two = '_2.pkl' if args.environment == 'standard' else '2.pkl'
        env_name = '' if args.environment == 'standard' else '_' + args.environment
        
        with open(cwd+'/PLP_data/copycat_64'+ env_name + '.pkl', 'rb') as handle:
            demos = pickle.load(handle)

        with open(cwd+'/PLP_data/copycat_64'+ env_name + two, 'rb') as handle:
            add_demos = pickle.load(handle)

        demos = join(demos, add_demos)
    except:
        with open(args.demo_path, 'rb') as handle:
            demos = pickle.load(handle)
    
    demos = {'pos': demos['pos'][:sum(demos['leng_pos'][:num_demos])], 
             'neg': demos['neg'][:sum(demos['leng_neg'][:num_demos])],
             'leng_neg': demos['leng_neg'][:num_demos], 
             'leng_pos': demos['leng_pos'][:num_demos]}

    input_demos = demos['pos']

    counter = 0
    for d in input_demos:
        s, a = get_state_action_representation(d)
        print("{}: {}: {}".format(counter,s,a))
        counter += 1
        

    ## Sample DSL file generation
    sm = SamplingMachine(input_demos, demos['neg'], use_DSL_features=True, 
                         info=add_info)
    
    import sys
    sys.exit()
    
    ###############################################
    
    ## Sample hierarchical visualization
    dsl_data = args.dsl_data
    sm = SamplingMachine(input_demos, demos['neg'], use_DSL_features=True, 
                         path_to_DSL_data=dsl_data, all_data=demos)

    num_cl = args.num_clusters
    lbls, nclust = sm.cluster(method="hierarchical", vis=True, info=add_info,
                              num_clusters=num_cl)
    
    
    ###############################################
    
    ## Sample test for the validity of clusters

    clustering = sm.interpret_clusters()
    score = sm.evaluate_clusters(clustering)

    #demos_pos, denos_neg = sm.sample_from_clusters(1.0)
    #cnt = 0
    #for demo in demos_pos:
    #    s, a = get_state_action_representation(demo)
    #    print("{}: {}: {}".format(cnt,s,a))
    #    cnt += 1
        
    #h_rep = sm.get_hidden_reps()
    #print_clusters(h_rep, lbls, nclust, show=True)
