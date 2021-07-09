import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns

# ------------------------- STATS ---------------------------------------------------------
def getDropoutStats(df1,df2, title):
    print('')
    print(title)
    print('Initial participants: {}'.format(len(df1)))
    print('Difference: {}'.format(len(df1) - len(df2)))
    print('Percentage: {}'.format(r(1-len(df2)/len(df1))))

    for c in df1.conditionType.unique().tolist():
        print('Percentage in condition {}: {}'.format(c,r(1-len(df2[df2.conditionType == c])/len(df1[df1.conditionType == c]))))

def print_pearson(group1, group2):
    r, p = stats.pearsonr( group1, group2)
    r = round(r, 4)
    p = round(p, 4)
    print("Pearson correlation: r = {} , p = {}".format(r,p))

def print_ttest(rews_group1, rews_group2):
    t_statistic, pval = stats.ttest_ind(rews_group1,rews_group2)
    print('t: {}, p_val: {}'.format(r(t_statistic), r(pval,digit=8)))

def print_whitney(rews_group1, rews_group2):
    u_statistic, pval = stats.mannwhitneyu(rews_group1,rews_group2)
    print('u: {}, p_val: {}'.format(r(u_statistic), r(pval,digit=8)))

def print_anova(dfs, key):
    values = []
    for df in dfs:
        values.append(df[key])
    print(stats.f_oneway(*values))
    print('')

def print_kruskal(dfs, key):
    values = []
    for df in dfs:
        values.append(df[key])
    print(stats.kruskal(*values))
    print('')

def getPTable(dfs, key):
    cols = ['cond']
    for df in dfs:
        cols.append(str(df.conditionType.iloc[0]))

    df_p = pd.DataFrame(columns = cols)
    for df1 in dfs:
        res = []
        for df2 in dfs:
            res.append(round(stats.ttest_ind(df1[key], df2[key])[1], 3))
        df_p.loc[len(df_p)] = [str(df1.conditionType.iloc[0])] + res

    return df_p

def getUTable(dfs, key):
    cols = ['cond']
    for df in dfs:
        cols.append(str(df.conditionType.iloc[0]))

    df_p = pd.DataFrame(columns = cols)
    for df1 in dfs:
        res = []
        for df2 in dfs:
            res.append(round(stats.mannwhitneyu(df1[key], df2[key])[1], 3))
        df_p.loc[len(df_p)] = [str(df1.conditionType.iloc[0])] + res

    return df_p


def print_statistics(dfs, key, title, parametric = True):
    print(title)
    print('')

    for df in dfs:
        print(' -- ' + str(df.conditionType.iloc[0]) + ' -- ')
        print('Mean: {}'.format(r(np.mean(df[key]))))
        print('Median: {}'.format(r(np.median(df[key]))))
        print('Std: {}'.format(r(np.std(df[key]))))
        print()

    print('')
    if(parametric):
        print('Anova')
        print_anova(dfs, key)
        print('T-table')
        print(getPTable(dfs, key))

    else:
        print('Kruskal Wallis Test')
        print_kruskal(dfs, key)
        print('U-table')
        print(getUTable(dfs, key))

    print('')
    #print_simple_barplots(dfs, key)
    print_simple_boxplot(dfs, key)


def print_parametric_info(dfs, df_valid, key, demo = False):
    values = []
    for df in dfs:
        print(df.conditionType.iloc[0])
        print('Shapiro: {}'.format(stats.shapiro(df[key])))

        plt.figure(len(values))
        weights = np.ones_like(df[key].tolist())/float(len(df[key].tolist()))
        if demo:
            plt.hist(df[key], weights=weights, bins = 2)
        else:
            plt.hist(df[key], weights=weights)
        plt.title(df.conditionType.iloc[0])
        plt.ylabel('Proportion')
        plt.xlabel('Pair Agreement')
        plt.ylim((0,1))
        #stats.probplot(df[key], plot=plt)

        values.append(df[key])
        print('')

    print('general')
    plt.figure(len(values))
    weights = np.ones_like(df_valid[key].tolist())/float(len(df_valid[key].tolist()))
    if demo:
        plt.hist(df_valid[key], weights=weights, bins = 2)
    else:
        plt.hist(df_valid[key], weights=weights)

    print('Shapiro: {}'.format(stats.shapiro(df_valid[key])))
    print(stats.fligner(*values))


def print_simple_barplots(dfs, key):
    labels = []
    means = []
    errors = []
    for df in dfs:
        labels.append(df.conditionType.iloc[0])
        means.append(df[key].mean())
        errors.append(1.96*df[key].std()/np.sqrt(len(df)))

    plt.bar(x=range(len(dfs)), height=means, yerr=errors, tick_label= labels)

def print_simple_boxplot(dfs, key):
    labels = []
    data = []
    for df in dfs:
        data.append(df[key].tolist())
        labels.append(df.conditionType.iloc[0])

    plt.boxplot(data, labels=labels)


def r(number, digit=4):
    return round(number,digit)

def print_statistics2(df1, df2, key):
    print(' -- df1 -- ')
    print('Median: {}'.format(r(np.median(df1[key]))))
    print('Mean: {}'.format(r(np.mean(df1[key]))))
    print('Std: {}'.format(r(np.std(df1[key]))))
    print('(${}\% \; (M= {}\%, SD = {}\%)$)'.format( r(np.median(df1[key])) , r(np.mean(df1[key])), r(np.std(df1[key]))))
    print()

    print(' -- df2 -- ')
    print('Median: {}'.format(r(np.median(df2[key]))))
    print('Mean: {}'.format(r(np.mean(df2[key]))))
    print('Std: {}'.format(r(np.std(df2[key]))))
    print('(${}\% \; (M= {}\%, SD = {}\%)$)'.format( r(np.median(df2[key])) , r(np.mean(df2[key])), r(np.std(df2[key]))))
    print()

    print(' -- test-- ')
    print_ttest(df1[key], df2[key])
    print_whitney(df1[key], df2[key])
    print()

# ------------------------- print learning curves ---------------------------------------------------------
def print_LC(dfs, key, title='', ylabel = 'PA', xlabel = 'Trial', measure='median'):

    if measure == 'mean':
        index = 1

        for df, label, col in zip(dfs, ['Training', 'No-Training'], ["#4990bc", "#dc551b"]):
            means = np.mean(df[key].tolist(), axis = 0)
            errors = 1.96 * np.std(df[key].tolist(), axis = 0)/np.sqrt(len(df))

            #plt.errorbar(index + np.arange(len(df[key].iloc[0])), means, yerr=errors, fmt='-o', label = df.conditionType.iloc[0])
            plt.errorbar(index + np.arange(len(df[key].iloc[0])), means, yerr=errors, fmt='-o', label = label, color=col)
            index += 0.1

    if measure == 'median':
        index = 0

        for df in dfs:
            means = np.median(df[key].tolist(), axis = 0)
            errors = [[],[]]

            for i in range(len(df[key].iloc[0])):
                lower, upper = get_median_cofindence(np.array(df[key].tolist())[:,i])
                errors[0].append(lower)
                errors[1].append(upper)

            plt.errorbar(index + np.arange(len(df[key].iloc[0])), means, yerr=errors, fmt='-o', label = df.conditionType.iloc[0])
            index += 0.1

    plt.legend()
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

def get_median_cofindence(dataset):
        alpha = 5.0
        scores = []
        for i in range(100):
        	# bootstrap sample
        	indices = np.random.randint(0, len(dataset), 1000)
        	sample = dataset[indices]
        	# calculate and store statistic
        	statistic = np.mean(sample)
        	scores.append(statistic)

        # calculate lower percentile (e.g. 2.5)
        lower_p = alpha / 2.0
        # retrieve observation at lower percentile
        lower = max(0.0, np.percentile(scores, lower_p))

        # calculate upper percentile (e.g. 97.5)
        upper_p = (100 - alpha) + (alpha / 2.0)
        # retrieve observation at upper percentile
        upper = min(1.0, np.percentile(scores, upper_p))

        return lower, upper


# ------------------------- BAR PLOTS ---------------------------------------------------------
def create_barplot(df1, df2, key, bins):

    exp = df1[key]
    cont = df2[key]
    exp_norm = plt.hist([exp, cont], bins, histtype='bar')[0][0]
    cont_norm = plt.hist([exp, cont], bins, histtype='bar')[0][1]
    exp_norm = exp_norm /sum(exp_norm)
    cont_norm = cont_norm /sum(cont_norm)

    exp_err_up = []
    exp_err_down = []
    for x in exp_norm:
        sd = np.sqrt(x*(1-x))
        err = sd/np.sqrt(len(df1))
        err = 1.96*err
        exp_err_up.append(err)
        if err > x:
            err = x
        exp_err_down.append(err)

    cont_err_up = []
    cont_err_down = []
    for x in cont_norm:
        sd = np.sqrt(x*(1-x))
        err = sd/np.sqrt(len(df2))
        err = 1.96*err
        cont_err_up.append(err)
        if err > x:
            err = x
        cont_err_down.append(err)

    print(bins)
    dif = abs(bins[0]-bins[1])/2
    pos = []
    for x in bins[0:len(bins)-1]:
        pos.append(x + dif)

    print(pos)
    plt.clf()
    plt.bar( pos, exp_norm, yerr=[exp_err_down, exp_err_up],  width= dif*0.7, align='edge', label="Training", capsize=4, error_kw=dict(lw=1), color="#4990bc")
    plt.bar( pos, cont_norm, yerr=[cont_err_down, cont_err_up], width= dif*-0.7, align='edge', label="No Training", capsize=4, error_kw=dict(lw=1), color= "#dc551b")

    plt.xlim((min(bins), max(bins)))
    plt.xticks(bins)
    plt.legend()



def create_barplot2(df, cond1, cond2, key, bin_number):

    exp  = (df[df.conditionType == cond1][key]/10).tolist()
    cont = (df[df.conditionType == cond2][key]/10).tolist()
    bins = getBins(exp + cont, bin_number)

    exp_norm = plt.hist([exp, cont], bins, histtype='bar')[0][0]
    cont_norm = plt.hist([exp, cont], bins, histtype='bar')[0][1]
    exp_norm = exp_norm /sum(exp_norm)
    cont_norm = cont_norm /sum(cont_norm)

    exp_err_up = []
    exp_err_down = []
    for x in exp_norm:
        sd = np.sqrt(x*(1-x))
        err = sd/np.sqrt(len(exp))
        err = 1.96*err
        exp_err_up.append(err)
        if err > x:
            err = x
        exp_err_down.append(err)

    cont_err_up = []
    cont_err_down = []
    for x in cont_norm:
        sd = np.sqrt(x*(1-x))
        err = sd/np.sqrt(len(cont))
        err = 1.96*err
        cont_err_up.append(err)
        if err > x:
            err = x
        cont_err_down.append(err)

    dif = abs(bins[0]-bins[1])/2
    pos = []
    for x in bins[0:len(bins)-1]:
        pos.append(x + dif)

    plt.subplot(1,3,3).cla()
    plt.bar( pos, exp_norm, yerr=[exp_err_down, exp_err_up],  width= dif*0.7, align='edge', label=cond1, capsize=4, error_kw=dict(lw=1))
    plt.bar( pos, cont_norm, yerr=[cont_err_down, cont_err_up], width= dif*-0.7, align='edge', label=cond2, capsize=4, error_kw=dict(lw=1))

    plt.xlim((min(bins), max(bins)))
    plt.xticks(bins)
    plt.legend()


# takes arrary and number of bins and suggests bin distribution
def getBins(arr, number):
    # extend min max
    mini = np.floor(min(arr))
    maxi = np.ceil(max(arr))

    # get scale differences
    diff = abs(maxi - mini)
    diff_scale =  number * (np.ceil(diff/number) % diff)
    offset = (diff_scale - diff)

    # prject scale differences on real scale
    if (offset % 2) == 0:
        mini_scale = mini - offset/2
        maxi_scale = maxi + offset/2
    else:
        mini_scale = mini - np.floor(offset/2)
        maxi_scale = maxi + np.floor(offset/2) + 1

    # suggest bins
    step = diff_scale/number
    print(mini_scale, maxi_scale+1, step)
    bins = np.arange(mini_scale, maxi_scale+1, step)
    pos = np.arange(mini_scale+step/2, maxi_scale, step)

    print(min(arr))
    print(max(arr))
    print(bins)
    print(pos)

    return bins
