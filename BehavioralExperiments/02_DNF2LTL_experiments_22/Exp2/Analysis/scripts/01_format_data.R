# 0 Packages ----------------------------------------------------------------
library(psych)
library(ggplot2)
library(scales)
library(ggpubr)
library(Hmisc)

# 1 Import ----------------------------------------------------------------
rm(list = ls())

participant_frame <- read.csv('../data/participant_frame.csv')
trial_frame <- read.csv('../data/trial_frame.csv')

# cast
participant_frame$totalTime <- as.numeric(participant_frame$totalTime)
participant_frame$age <- as.numeric(participant_frame$age)
trial_frame$FSQ <- as.numeric(trial_frame$FSQ)
trial_frame$stoppedAfterBestValue <- as.numeric(trial_frame$stoppedAfterBestValue == 'True')
trial_frame$encounteredBestValue <- as.numeric(trial_frame$encounteredBestValue == 'True')

# Filter for finished 
pf_failed_quiz <- subset(participant_frame, status == 6 & (RT_attemptsQuiz == 3 | MG_attemptsQuiz == 3))
pf_valid <- subset(participant_frame, status == 3)

# Filter trial frame accordingly
tf_valid <- subset(trial_frame, pid %in% pf_valid$pid)
