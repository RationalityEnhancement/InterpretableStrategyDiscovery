library(rcompanion)
library(psych)
library(car)
library(ggplot2)
library(scales)

"
 We have:
  Betweeen factor: condition
  dependent Variable: pa_mean
"


# IMPORT --------------------------------------------------------------
setwd("~/Documents/Dev/Interpretability/InterpretableStrategyDiscovery/Procedural_Instructions_Experiments/analysis/Experiment1")
df_all <- read.csv('data/dataframe_complete.csv')


# Exclusion  --------------------------------------------------------------
# early quitter
df_finished <- df_all[df_all$status != 6,]

# cast variables
df_finished$totalTime <- as.numeric(df_finished$totalTime)
table(df_finished$conditionType)

# exclusion criteria
df_valid <- df_finished[(df_finished$cheatTrials <= 5),]
table(df_finished$conditionType)

# Descriptive------------------------------------------------------------------
sum(grepl('fe', df_finished[ df_finished$gender != '-',]$gender, ignore.case = TRUE)) # female?
summary(strtoi(df_finished$age)) # age
summary(as.numeric(df_finished$totalTime)) #duration
summary(as.numeric(df_finished$attemptsQuiz)) #attempts
summary(as.numeric(df_finished$attemptsQuiz2)) #attempts2
summary(as.numeric(subset(df_finished, condition == 1)$cheatTrials)) # cheat trials
table(df_valid$conditionType)

# Click Agreement ------------------------------------------------------------------

describeBy(df_valid$pa_mean, df_valid$conditionType, digits=3, mat=TRUE)
boxplot(pa_mean ~ conditionType, data=df_valid)

# normality
plotNormalHistogram(df_valid$pa_mean, breaks = 50)
qqPlot(df_valid$pa_mean)
shapiro.test(df_valid$pa_mean)

par(mfrow=c(2,1))
plotNormalHistogram(df_valid[df_valid$condition == 0,]$pa_mean, breaks = 5, prob=TRUE)
plotNormalHistogram(df_valid[df_valid$condition == 1,]$pa_mean, breaks = 5, prob=TRUE)
par(mfrow=c(1,1))

# 
wilcox.test(pa_mean ~ conditionType, data=df_valid)

# plot boxplot
ggplot(df_valid, aes(x=conditionType, y=pa_mean, fill=conditionType)) +
  geom_boxplot(alpha=0.9) + 
  scale_fill_manual(values=c("#6ea7ca", "#e87442", '#3fba4a')) +
  scale_y_continuous(labels = percent, minor_breaks = seq(0 , 0, 0)) +
  theme(legend.position="none") +
  ylab('Click agreement') + 
  xlab('Condition') + 
  scale_x_discrete(labels=c("flowchart" = "flowchart", "instructions1" = "instructions")) +
  theme(text = element_text(size = 15))

# plot violine
ggplot(df_valid, aes(x=conditionType, y=pa_mean, fill=conditionType)) +
  geom_violin(alpha = 0.5) +
  scale_fill_manual(values=c("#6ea7ca", "#e87442")) +
  geom_boxplot(width = 0.1, color = 'black', fill = c("#6ea7ca", "#e87442"), alpha = 1) + 
  scale_y_continuous(labels = percent, minor_breaks = seq(0 , 0, 0)) +
  theme(legend.position="none") +
  ylab('Click agreement') + 
  xlab('Condition') + 
  ggtitle('Click agreement per group')+
  scale_x_discrete(labels=c("flowchart" = "flowchart", "instructions1" = "procedural-instructions")) +
  theme(text = element_text(size = 34), plot.title = element_text(hjust = 0.5, size = 31))

