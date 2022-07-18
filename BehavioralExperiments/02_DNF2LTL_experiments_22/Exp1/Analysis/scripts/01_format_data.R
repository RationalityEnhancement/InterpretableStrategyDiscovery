library(rcompanion)
library(psych)
library(car)
library(ggplot2)
library(scales)
library(effsize)
library(ggpubr)

# IMPORT --------------------------------------------------------------

# reset workspace
rm(list = ls())

# load data
df_all <- read.csv('../data/dataframe_complete.csv')


# Exclusion  --------------------------------------------------------------

# early quitter
df_finished <- df_all[df_all$status != 6,]

# cast variables
df_finished$totalTime <- as.numeric(df_finished$totalTime)

# exclusion criteria
df_valid <- df_finished[(df_finished$cheatTrials <= 5),]


# Create a Trial Frame --------------------------------------------------------------
readArrayColumn <- function(col, ncol){
  res<- data.frame(matrix(ncol=ncol,nrow=0 ))
  for (el in col) {
    res[nrow(res) + 1,] = scan(text=substr(el, 2, nchar(el)-1), sep=',')
  }
  res
}

trial_frame <- data.frame('pid' = rep(df_valid$pid, each=10),
                          'index'= rep(1:10, times=nrow(df_valid)),
                          'condition'= rep(df_valid$condition, each=10),
                          'pa' =  as.vector(t(readArrayColumn(df_valid$pa_complete, 10))),
                          'expectedScore' =  as.vector(t(readArrayColumn(df_valid$expectedScores, 10)))
                          )

rm(readArrayColumn)


# Import results from previous experiment for comparison ------------------

# load results from old experiment for comparison
df_valid_previousexp <- read.csv('../data/previouswork/df_valid.csv')

# take relevant subset
df_valid_previousexp <- df_valid_previousexp[c('WorkerId', 'condition', 'condition_type',
                                               'pair_agreement', 'mean_rew_exp')]
# rename
names(df_valid_previousexp) <- c('pid', 'condition', 'conditionType', 'pa_mean', 'expectedScoreMean')
df_valid_previousexp <- subset(df_valid_previousexp, conditionType == "control")
df_valid_previousexp$condition = 2

# combine with new data
df_combined <- rbind(
  df_valid[c('pid', 'condition', 'conditionType', 'pa_mean', 'expectedScoreMean')],
  df_valid_previousexp
  )

rm(df_valid_previousexp)

