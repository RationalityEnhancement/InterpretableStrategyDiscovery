# 0 Packages ----------------------------------------------------------------
library(psych)
library(ggplot2)
library(scales)
library(ggpubr)

# 1 Import ----------------------------------------------------------------
participant_frame <- read.csv('data/participant_frame.csv')
trial_frame <- read.csv('data/trial_frame.csv')

# cast
participant_frame$totalTime <- as.numeric(participant_frame$totalTime)
participant_frame$age <- as.numeric(participant_frame$age)
trial_frame$FSQ <- as.numeric(trial_frame$FSQ)

# Filter for finished 
pf_failed_quiz <- subset(participant_frame, status == 6 & (RT_attemptsQuiz == 3 | MG_attemptsQuiz == 3))
pf_valid <- subset(participant_frame, status == 3)

# Filter trial frame accordingly
tf_valid <- subset(trial_frame, pid %in% pf_valid$pid)


# 2 Descriptive Statistics-------------------------------------------------------------

# participant numbers
table(participant_frame$conditionType)
table(pf_failed_quiz$conditionType)
table(pf_valid$conditionType)

# gender
pf_valid$female <- grepl('fe', pf_valid[pf_valid$gender != '-',]$gender, ignore.case = TRUE) # female?
table(pf_valid$female, pf_valid$conditionType)

#age
summary(pf_valid$age)

#time
summary(pf_valid$totalTime)

# 3 FSQ Analysis ----------------------------------------------------------------

# Aggregate mean FSQ per participant
mean_FSQ <- aggregate(.~ pid + conditionType + task, tf_valid[c('pid', 'FSQ', 'conditionType', 'task')], mean)

# FSQ stats
aggregate(FSQ ~ conditionType + task, mean_FSQ, mean)
aggregate(FSQ ~ conditionType + task, mean_FSQ, median)

# statistical test roadtrip
mean_FSQ <- aggregate(.~ pid + conditionType, subset(tf_valid, task =='roadtrip')[c('pid', 'FSQ', 'conditionType')], mean)
wilcox.test(FSQ ~ conditionType, mean_FSQ, alternative='less')

# statistical test mortgage
mean_FSQ <- aggregate(.~ pid + conditionType, subset(tf_valid, task =='mortgage')[c('pid', 'FSQ', 'conditionType')], mean)
wilcox.test(FSQ ~ conditionType, mean_FSQ, alternative='less')



# 4 Score----------------------------------------------------------------

# Aggregate mean FSQ per participant
mean_score <- aggregate(.~ pid + conditionType + task, tf_valid[c('pid', 'score', 'conditionType', 'task')], mean)

# score
mes <- median
aggregate(score ~ task, mean_score, mes)
aggregate(score ~ conditionType + task, mean_score, mes)


# statistical test roadtrip
mean_score <- aggregate(.~ pid + conditionType, subset(tf_valid, task =='roadtrip')[c('pid', 'score', 'conditionType')], mean)
wilcox.test(score ~ conditionType, mean_score, alternative='less')
cohens_d(score ~ conditionType, data =mean_score)

# statistical test mortgage
mean_score <- aggregate(.~ pid + conditionType, subset(tf_valid, task =='mortgage')[c('pid', 'score', 'conditionType')], mean)
wilcox.test(score ~ conditionType, mean_score, alternative='less')
cohens_d(score ~ conditionType, data =mean_score)


# 5 Plots ----------------------------------------------------------------
colors <- c("#3977AF", "#EF8536")
fontsize <- 25
fontsize_title <- 25
cond_names <- c('control', 'supported by \n decision aid')
  
# first plot mean FSQ per group
df <- aggregate(FSQ ~ pid + conditionType, subset(tf_valid, task=='roadtrip'), mean)
p1 <- ggplot(df, aes_string(x='conditionType', y='FSQ', fill='conditionType')) +
  geom_violin(alpha = 0.5) +
  scale_fill_manual(values=colors) +
  geom_boxplot(width = 0.1, color = 'black', fill = colors, alpha = 1) + 
  scale_y_continuous(minor_breaks = seq(0 , 0, 0), labels = percent) +
  scale_x_discrete(labels = cond_names) +
  theme(legend.position="none") +
  theme(text = element_text(size = fontsize),
        plot.title = element_text(hjust = 0.5, size = fontsize_title), 
        axis.text=element_text(size=fontsize)) + 
  ylab('Mean FSQ') + 
  xlab('') + 
  ggtitle('Far-sightedness in the \n Roadtrip task')

# first plot mean FSQ per group
df <- aggregate(FSQ ~ pid + conditionType, subset(tf_valid, task=='mortgage'), mean)
p2 <- ggplot(df, aes_string(x='conditionType', y='FSQ', fill='conditionType')) +
  geom_violin(alpha = 0.5) +
  scale_fill_manual(values=colors) +
  geom_boxplot(width = 0.1, color = 'black', fill = colors, alpha = 1) + 
  scale_x_discrete(labels = cond_names) +
  scale_y_continuous(minor_breaks = seq(0 , 0, 0), labels = percent) +
  theme(legend.position="none") +
  theme(text = element_text(size = fontsize),
        plot.title = element_text(hjust = 0.5, size = fontsize_title), 
        axis.text=element_text(size=fontsize)) + 
  ylab('Mean FSQ') + 
  xlab('') + 
  ggtitle('Far-sightedness in the \n Mortgage task')

# second plot mean Score per group
df <- aggregate(score ~ pid + conditionType, subset(tf_valid, task=='roadtrip'), mean)
p3 <- ggplot(df, aes_string(x='conditionType', y='score', fill='conditionType')) +
  geom_violin(alpha = 0.5) +
  scale_fill_manual(values=colors) +
  geom_boxplot(width = 0.1, color = 'black', fill = colors, alpha = 1) + 
  scale_x_discrete(labels = cond_names) +
  scale_y_continuous(minor_breaks = seq(0 , 0, 0)) +
  theme(legend.position="none") +
  theme(text = element_text(size = fontsize),
        plot.title = element_text(hjust = 0.5, size = fontsize_title), 
        axis.text=element_text(size=fontsize)) + 
  ylab('Mean score') + 
  xlab('') + 
  ggtitle('Performance in the \n Roadtrip task')


# third plot mean Score per group
df <- aggregate(score ~ pid + conditionType, subset(tf_valid, task=='mortgage'), mean)
p4 <- ggplot(df, aes_string(x='conditionType', y='score', fill='conditionType')) +
  geom_violin(alpha = 0.5) +
  scale_fill_manual(values=colors) +
  geom_boxplot(width = 0.1, color = 'black', fill = colors, alpha = 1) + 
  scale_y_continuous(minor_breaks = seq(0 , 0, 0), labels = percent) +
  scale_x_discrete(labels = cond_names) +
  theme(legend.position="none") +
  theme(text = element_text(size = fontsize),
        plot.title = element_text(hjust = 0.5, size = fontsize_title), 
        axis.text=element_text(size=fontsize)) + 
  ylab('Optimal choices') + 
  xlab('') + 
  ggtitle('Performance in the \n Mortgage task')

pdf('plots/Exp2_combined.pdf', width = 14, height = 12)
ggarrange(p2, p1, p4, p3, labels = c("a", "b", "c", "d"), ncol = 2, nrow = 2, font.label = list(size = fontsize))
dev.off()





# 6 CA Analysis ----------------------------------------------------------------

# Aggregate mean CA per participant
mean_CA <- aggregate(.~ pid + conditionType + task, tf_valid[c('pid', 'CA', 'conditionType', 'task')], mean)

# CA stats
aggregate(CA ~ conditionType + task, mean_CA, mean)
aggregate(CA ~ conditionType + task, mean_CA, median)

# statistical test roadtrip
mean_CA <- aggregate(.~ pid + conditionType, subset(tf_valid, task =='roadtrip')[c('pid', 'CA', 'conditionType')], mean)
wilcox.test(CA ~ conditionType, mean_CA, alternative='less')

# statistical test mortgage
mean_CA <- aggregate(.~ pid + conditionType, subset(tf_valid, task =='mortgage')[c('pid', 'CA', 'conditionType')], mean)
wilcox.test(CA ~ conditionType, mean_CA, alternative='less')

# correlation
cor.test(tf_valid$FSQ, tf_valid$CA)

