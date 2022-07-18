
source('01_format_data.R')

# 1 Descriptive Statistics-------------------------------------------------------------

# participant numbers
table(participant_frame$conditionType)
table(pf_failed_quiz$conditionType)
table(pf_valid$conditionType)

# gender
pf_valid$female <- grepl('fe', pf_valid[pf_valid$gender != '-',]$gender, ignore.case = TRUE) # female?
table(pf_valid$female, pf_valid$conditionType)

#age
summary(pf_valid$age)
describeBy(pf_valid$age, pf_valid$conditionType)

#time
describeBy(pf_valid$totalTime, pf_valid$conditionType)


# 2 FSQ Analysis ----------------------------------------------------------------

# Aggregate mean FSQ per participant
mean_FSQ <- aggregate(.~ pid + conditionType + task, tf_valid[c('pid', 'FSQ', 'conditionType', 'task')], mean)

# FSQ stats
aggregate(FSQ ~ conditionType + task, mean_FSQ, mean)
aggregate(FSQ ~ conditionType + task, mean_FSQ, median)

# statistical test roadtrip
mean_FSQ <- aggregate(.~ pid + conditionType, subset(tf_valid, task =='roadtrip')[c('pid', 'FSQ', 'conditionType')], mean)
wilcox.test(FSQ ~ conditionType, mean_FSQ, alternative='less')

# effect size: probability-based measure A
((54*55)-741)/(54*55)

# statistical test mortgage
mean_FSQ <- aggregate(.~ pid + conditionType, subset(tf_valid, task =='mortgage')[c('pid', 'FSQ', 'conditionType')], mean)
wilcox.test(FSQ ~ conditionType, mean_FSQ, alternative='less')

# effect size: probability-based measure A
((54*55)-658)/(54*55)

# 3 Score----------------------------------------------------------------

# Aggregate mean FSQ per participant
mean_score <- aggregate(.~ pid + conditionType + task, tf_valid[c('pid', 'score', 'conditionType', 'task')], mean)

# score
mes <- median
aggregate(score ~ task, mean_score, mes)
aggregate(score ~ conditionType + task, mean_score, mes)


# statistical test roadtrip
mean_score <- aggregate(.~ pid + conditionType, subset(tf_valid, task =='roadtrip')[c('pid', 'score', 'conditionType')], mean)
wilcox.test(score ~ conditionType, mean_score, alternative='less')

# effect size: probability-based measure A
((54*55)-995)/(54*55)

# statistical test mortgage
mean_score <- aggregate(.~ pid + conditionType, subset(tf_valid, task =='mortgage')[c('pid', 'score', 'conditionType')], mean)
wilcox.test(score ~ conditionType, mean_score, alternative='less')

# effect size: probability-based measure A
((54*55)-654)/(54*55)


# 4 CA Analysis ----------------------------------------------------------------

# Aggregate mean CA per participant
mean_CA <- aggregate(.~ pid + conditionType + task, tf_valid[c('pid', 'CA', 'conditionType', 'task')], mean)

# CA stats
aggregate(CA ~ conditionType + task, mean_CA, mean)
aggregate(CA ~ conditionType + task, mean_CA, median)

# statistical test roadtrip
mean_CA <- aggregate(.~ pid + conditionType, subset(tf_valid, task =='roadtrip')[c('pid', 'CA', 'conditionType')], mean)
wilcox.test(CA ~ conditionType, mean_CA, alternative='less')

# effect size: probability-based measure A
((54*55)-838)/(54*55)

# statistical test mortgage
mean_CA <- aggregate(.~ pid + conditionType, subset(tf_valid, task =='mortgage')[c('pid', 'CA', 'conditionType')], mean)
wilcox.test(CA ~ conditionType, mean_CA, alternative='less')

# effect size: probability-based measure A
((54*55)-659)/(54*55)

# correlation
cor.test(tf_valid$FSQ, tf_valid$CA)



# 5 Analysis of correct stopping ----------------------------------------

# add if in the previous trial the best value was encountered
df <- subset(trial_frame, task == 'roadtrip')
df$encounteredBestValue_prev <- Lag(df$encounteredBestValue, 1)

# in first round no has previously encountered the best value
df[df$trial == 0, ]$encounteredBestValue_prev <- 0
aggregate(FSQ ~ encounteredBestValue_prev, df, mean)

# test 
df_mean <- aggregate(FSQ ~ pid + encounteredBestValue_prev, df, mean)
wilcox.test(FSQ ~ encounteredBestValue_prev, df_mean)

# effect size: probability-based measure A
((112*89)-3203)/(112*89)


# 6 Regression Model ------------------------------------------------------

library(lme4)
library(lmerTest)

df_m <- subset(tf_valid, task == 'mortgage')
df_r <- subset(tf_valid, task == 'roadtrip')

aggregate(score ~ trial + conditionType, df_m, mean)

# -- FSQ
model <- lmer('FSQ ~ trial * conditionType + (1|pid)', df_m)
summary(model)

model <- lmer('FSQ ~ trial * conditionType + (1|pid)', df_r)
summary(model)

# -- Performance
model <- glmer('score ~ trial * conditionType + (1|pid)', df_m, family = binomial)
summary(model)

model <- lmer('score ~ trial * conditionType + (1|pid)', df_r)
summary(model)


# 7 Analysis of alternative decision strategy ----------------------------


df <- subset(trial_frame, FSQ < 1)
res_first_click <- c()
res_second_click <- c()

for(clicks in df$levelOfClicks){
  
  click <- substr(clicks, 2, 2) # first click
  if(click == "]"){
    res_first_click <- c(res_first_click, c(-1))
    
  } else {
    res_first_click <- c(res_first_click, click)
    
    click <- substr(clicks, 5, 5) # second click
    if(click == ""){
      res_second_click <- c(res_second_click, c(-1))
      
    } else {
      res_second_click <- c(res_second_click, click)
    }
    
  }
}

print(prop.table(table(res_first_click)))
print(prop.table(table(res_second_click)))


