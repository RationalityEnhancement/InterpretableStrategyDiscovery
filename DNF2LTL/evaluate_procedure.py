import random
import os
import argparse
import numpy as np
import pickle
import copy
import scipy.integrate
from RL2DT.strategy_demonstrations import make_modified_env as make_env
from scipy.special import softmax
from RL2DT.MCRL.modified_mouselab_new import TrialSequence
from RL2DT.PLP.DSL import *
from progress.bar import ChargingBar
from scipy.stats import norm as Normal
from scipy.special import beta as Beta
from hyperparams import ITERS
from functools import lru_cache

ALPHA = 1
BETA = 7

class ProceduralStrategy(object):
    """
    Class handling a procedural description with elements particular to the linear
    temporal logic formulas.
    """
    def __init__(self, text_formula):
        self.strategies = text_formula.split(' AND NEXT ')
        self.current_index = 0
        self.strategies = self.strategies[:-1] + self.strategies[-1].split('\n\n')
        self.current_index = 0
        st_unl = self.strategies[0].split(" UNLESS ")
        self.current = st_unl[0]
        st_unt = self.current.split(" UNTIL ")
        self.current = st_unt[0]
        self.unless = st_unl[1] if len(st_unl) > 1 else ""
        self.until = st_unt[1] if len(st_unt) > 1 else "ONE-STEP"
        self.steps = 0
        
    def next(self):
        self.current_index += 1
        loop = False
        
        if self.current_index == len(self.strategies):
            raise Exception("The end of the strategy")
        
        if "LOOP" in self.strategies[self.current_index]:
            loop = self.strategies[self.current_index][10:]
            lp_unl = loop.split(" UNLESS ")
            loop_formula = lp_unl[0]
            for i, s in enumerate(self.strategies):
                clean_s = s[2:-2] if s[:2] == "((" else s
                if loop_formula in clean_s:
                    self.current_index = i
                    break
            self.unless = lp_unl[1] if len(lp_unl) > 1 else ""
            loop = True
            
        st_unl = self.strategies[self.current_index].split(" UNLESS ")
        self.current = st_unl[0]
        st_unt = self.current.split(" UNTIL ")
        self.current = st_unt[0]
        
        if not(loop): self.unless = st_unl[1] if len(st_unl) > 1 else ""
        self.until = st_unt[1] if len(st_unt) > 1 else "ONE-STEP"
        self.steps = 0
        
    def until_condition_satisfied(self, state, unobs):
        if "ONE-STEP" in self.until:
            return self.steps != 0
        elif "STOPS APPLYING" in self.until:
            return not(any([eval('lambda st, act : ' + self.current)(state, a) 
                        for a in unobs]))
        else:
            return all([eval('lambda st, act : ' + self.until)(state, a) 
                        for a in unobs])
        
    def applies(self, state, unobs):
        return any([eval('lambda st, act : ' + self.current)(state, a) 
                        for a in unobs])
    
    def block_strategy(self, state, unobs):
        if self.unless == '':
            return False
        else:
            vals = [eval('lambda st, act : ' + self.unless)(state, a) 
                    for a in unobs]
            vals = [bool(v) for v in vals]
            return any(vals)
        
    def step(self):
        self.steps += 1


def return_allowed_acts(strategy_str, state, unobserved_nodes):
    """
    Output an action congruent with a logical formula.

    Parameters
    ----------
    (See take_action)
    strategy_str : str
    state : IHP.modified_mouselab.Trial
    unobserved_nodes : [ int ]
        
    Returns
    -------
    allowed : [ int ]
        List of actions allowed by the input formula strategy_str
    """
    actions = {a: eval('lambda st, act : ' + strategy_str)(state, a) 
               for a in unobserved_nodes}
    allowed = [a for a in actions.keys() if actions[a] == True]
    if allowed == []:
        return [0]
    return allowed

@lru_cache()
def compute_log_likelihood_demos(envs, pipeline, acts, formula, verbose):
    """
    Measure the likelihood of the data under the policy induced by the 
    procedural formula.
    
    Compute the proportion of optimal actions.
    
    Compute the marginal likelihood of the data under this policy. Both
    analytically and through numerical integration.
    
    A model is defined as a procedural description of the softmax policy. 
    
    Marginalizing is done over an Epsilon parameter with a Beta(ALPHA, BETA) 
    prior distribution.
    
    Posterior distribution is Epsilon * Uniform(non-allowed-actions(description)
    + (1-Epsilon) * Uniform(allowed_actions(description)).
    
    Parameters
    ----------
    envs : [ [ int ] ]
        List of environments encoded as rewards hidden under the nodes; index of
        the reward corresponds to the id of the node in the Mouselab MDP
    pipeline : [ ( [ int ], function ) ]
        List of parameters used for creating the Mouselab MDP environments: 
        branching of the MDP and the reward function for specifying the numbers
        hidden by the nodes. For each rollout.
    acts : [ [ int ] ]
        List of action sequences taken by people in each consecutive environment
    formula : str
        Procedural formula
        
    Returns
    -------
    log_likelihood : float
        Log likelihood for data under the policy induced by the procedural
        formula
    log_marginal : float
        Log marginal likelihood of the policy induced by the procedural formula
    opt_score : float
        Proportion of actions taken according to the allowed actions of the  
        formula-induced policy
    mean_len : float
        Mean length of human demonstrations
    1-Epsilon : float
        Measure of fit of the procedural description with respect to the data;
        because of the particular form of the posterior, equal to opt_score
    """
    envs = [list(e) for e in envs]
    pipeline = [(list(pipeline[0]), pipeline[1])]
    acts = [list(p) for p in acts]
    
    optimal_acts = 0
    total_len = 0
    log_likelihood = 0
    total_likelihood, optimal_likelihood = 0, 0
    ll_allowed, ll_not_allowed = 0, 0
    epsilon_power, _1_epsilon_power = 0, 0
    if verbose: bar = ChargingBar('Computing people data', max=len(envs))
    for ground_truth, actions in zip(envs, acts):
        env = TrialSequence(1, pipeline, ground_truth=[ground_truth]).trial_sequence[0]
        new_iter = True
        formulas = formula.split('\n\nOR\n\n')
        strategies = []
        for f in formulas:
            strategies.append(ProceduralStrategy(f))
            
        for action in actions:
            total_len += 1
            unobserved_nodes = env.get_unobserved_nodes()
            unobserved_node_labels = [node.label for node in unobserved_nodes]
                        
            if new_iter:
                applicable_strategies = [s for s in strategies 
                                         if s.applies(env, unobserved_node_labels)]
            if applicable_strategies == []:
                #No planning strategy
                allowed_acts = [0]
            else:
                allowed_acts = []
            untils = {s: s.until_condition_satisfied(env, unobserved_node_labels) 
                      for s in applicable_strategies}
            
            if any(list(untils.values())):
                applicable_strategies = [s for s in applicable_strategies 
                                         if untils[s]]
                try:
                    for s in applicable_strategies:
                        s.next()
                        if not(s.block_strategy(env, unobserved_node_labels)):
                             allowed_acts += return_allowed_acts(s.current, 
                                                                 env,
                                                                 unobserved_node_labels)
                        else:
                            allowed_acts = [0]
                        s.step()
                except:
                    allowed_acts = [0]
            else:
                for s in applicable_strategies:
                    if not(s.block_strategy(env, unobserved_node_labels)):
                        allowed_acts += return_allowed_acts(s.current, 
                                                            env,
                                                            unobserved_node_labels)
                    else:
                        allowed_acts = [0]
                    s.step()
            
            if formula == "None": allowed_acts = []
            allowed_acts = list(set(allowed_acts))
            allowed_distr = {a: 1/float(len(allowed_acts)) for a in allowed_acts}
            not_allowed_acts = [a for a in env.node_map.keys()
                                if a not in allowed_acts]
            not_allowed_distr = {a: 1/float(len(not_allowed_acts)) 
                                 for a in not_allowed_acts}
            if action in allowed_acts:
                optimal_acts += 1
                ll_allowed += np.log(allowed_distr[action])
                total_likelihood += allowed_distr[action]
                _1_epsilon_power += 1
            else:
                ll_not_allowed += np.log(not_allowed_distr[action])
                epsilon_power += 1
            
            optimal_likelihood += allowed_distr[allowed_acts[0]]
            env.node_map[action].observe()
            new_iter = False
        if verbose: bar.next()
    if verbose: bar.finish()
    
    lik_per_action = total_likelihood / total_len
    opt_lik_per_action = optimal_likelihood / total_len
    opt_act_score = lik_per_action / opt_lik_per_action
    if verbose: print("Epsilon optimization...")
    ## Analytical solution to the global maximum in [0,1]
    if epsilon_power == 0:
        log_likelihood = ll_allowed
    elif _1_epsilon_power == 0:
        log_likelihood = ll_not_allowed
    else:
        epsilon = float(epsilon_power)/(epsilon_power + _1_epsilon_power)
        log_likelihood = epsilon_power*np.log(epsilon) + ll_not_allowed + \
                         _1_epsilon_power*np.log((1-epsilon)) + ll_allowed
                     
    ## computing epsilon maximizing likelihood * prior assuming prior is a 
    ## beta funciton, to later compute the marginal likelihood 
    nom = float(epsilon_power + ALPHA - 1)
    den = (epsilon_power + _1_epsilon_power + ALPHA + BETA - 2)
    epsilon = nom/den
    
    ## Assuming prior over epsilon g(epsilon) is Beta_funciton(ALPHA, BETA)
    ## Then using Laplace approximation to the marginal likelihood which is
    ## \int likelihood * g(epsilon); likelihood is E^K(1-E)^N * const, similar to g
    ## solved analytically                    
    reg = Beta(ALPHA, BETA)
    log_h = (epsilon_power+ALPHA-1)*np.log(epsilon) + ll_not_allowed + \
            (_1_epsilon_power+BETA-1)*np.log((1-epsilon)) + ll_allowed -\
            np.log(reg)
    h = np.exp(log_h)
    nom = (1-epsilon)**2 * epsilon**2
    den = epsilon**2 * (_1_epsilon_power+BETA-1) + \
          (1-epsilon)**2 * (epsilon_power+ALPHA-1)
    sigma_squared = nom/den if nom != 0 else 0
    normalizer = np.sqrt(2 * np.pi * sigma_squared)
    sigma = np.sqrt(sigma_squared)
    if epsilon_power+ALPHA-1 == 0:
        log_marginal = ll_allowed - np.log(reg) - np.log(_1_epsilon_power + BETA)
        marginal = np.exp(log_marginal)
    elif _1_epsilon_power+BETA-1 == 0:
        log_marginal = ll_not_allowed - np.log(reg) + np.log(1 / (epsilon_power+ALPHA))
        marginal = np.exp(log_marginal)
    else:
        marginal = h * normalizer * \
                   (Normal.cdf(1, epsilon, sigma) - Normal.cdf(0, epsilon, sigma))
        log_marginal = log_h + np.log(normalizer) + \
                       np.log(
                           Normal.cdf(1, epsilon, sigma) - Normal.cdf(0, epsilon, sigma))
    if verbose: 
        print("\nEpsilon: {}, h: {}, log_h: {}, ".format(epsilon, h, log_h) + \
              "sigma^2: {}, normalizer: {}, ".format(sigma_squared, normalizer) + \
              "log_marginal: {}, np.log(marginal): {}".format(log_marginal, 
                                                              np.log(marginal)))
    ## Numerical integration
    def integrand(x):
        eps = x**(epsilon_power+ALPHA-1) * (1-x)**(_1_epsilon_power+BETA-1)
        const = np.exp(ll_allowed+ll_not_allowed) / reg
        return eps * const
    result = scipy.integrate.quad(lambda x: integrand(x), 0, 1)
    if verbose: print("np.log(numerical_integral): {}".format(np.log(result[0])))
    if np.log(result[0]) != -np.inf and log_marginal == -np.inf:
        log_marginal = np.log(result[0])
        
    opt_score = optimal_acts/total_len
    mean_len = total_len/len(envs)
    return log_likelihood, log_marginal, opt_score, mean_len, 1-epsilon, opt_act_score
