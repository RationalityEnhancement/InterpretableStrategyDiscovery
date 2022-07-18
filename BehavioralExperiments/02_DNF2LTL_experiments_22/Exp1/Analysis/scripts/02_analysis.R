"
 We have:
  Betweeen factor: condition
  dependent Variable: pa_mean, trialTimeMean, expectedScoreMean, 
"

source('01_format_data.R')

# Participants statistics ------------------------------------------------------------------

sum(grepl('fe', df_finished[ df_finished$gender != '-',]$gender, ignore.case = TRUE)) # female?
summary(strtoi(df_finished$age)) # age
summary(df_finished$totalTime) #duration
summary(as.numeric(df_finished$attemptsQuiz)) #attempts
summary(as.numeric(df_finished$attemptsQuiz2)) #attempts2
summary(as.numeric(subset(df_finished, condition == 1)$cheatTrials)) # cheat trials

table(df_valid$conditionType)
table(df_finished$conditionType)


# Click Agreement ------------------------------------------------------------------

describeBy(df_combined$pa_mean, df_combined$conditionType, digits=3, mat=TRUE)
boxplot(pa_mean ~ conditionType, data=df_valid)

# normality
plotNormalHistogram(df_valid$pa_mean, breaks = 50)
qqPlot(df_valid$pa_mean)
shapiro.test(df_valid$pa_mean)

par(mfrow=c(2,1))
plotNormalHistogram(df_valid[df_valid$condition == 0,]$pa_mean, breaks = 5, prob=TRUE)
plotNormalHistogram(df_valid[df_valid$condition == 1,]$pa_mean, breaks = 5, prob=TRUE)
par(mfrow=c(1,1))

# test
wilcox.test(pa_mean ~ conditionType, data=df_valid, alternative ='less')

# effect size: probability-based measure A
((21*21)-137)/(21*21)

# comparison to control group:
wilcox.test(pa_mean ~ conditionType, data=subset(df_combined, conditionType != 'flowchart'), alternative='less')
((60*21)-59)/(60*21)

wilcox.test(pa_mean ~ conditionType, data=subset(df_combined, conditionType != 'instructions'), alternative='less')
((60*21)-93)/(60*21)

# Trial Duration ----------------------------------------------------------
describeBy(df_valid$trialTimeMean, df_valid$conditionType, digits=3, mat=TRUE)

# test
t.test(trialTimeMean ~ conditionType, data=df_valid, var.equal=TRUE, alternative ='greater')

# effect size
cohen.d(trialTimeMean ~ conditionType, data=df_valid)

# Expected score ----------------------------------------------------------
describeBy(df_combined$expectedScoreMean, df_combined$conditionType, digits=3, mat=TRUE)

# test
t.test(expectedScoreMean ~ conditionType, data=df_valid, var.equal=TRUE,  alternative ='less')

# effect size
cohen.d(expectedScoreMean ~ conditionType, data = df_valid)

# correlation
cor.test(df_valid$expectedScoreMean, df_valid$pa_mean)

# comparison to control group:
t.test(expectedScoreMean ~ conditionType, data=subset(df_combined, conditionType != 'flowchart'), var.equal=TRUE, alternative='less')
cohen.d(expectedScoreMean ~ conditionType, data = subset(df_combined, conditionType != 'flowchart'))

t.test(expectedScoreMean ~ conditionType, data=subset(df_combined, conditionType != 'instructions'), var.equal=TRUE, alternative='less')
cohen.d(expectedScoreMean ~ conditionType, data = subset(df_combined, conditionType != 'instructions'))

# Regression Model --------------------------------------------------------


library(lme4)
library(lmerTest)

# -- CA
model <- lmer('pa ~ index * condition + (1|pid)', trial_frame)
summary(model)

model <- lmer('pa ~ index + (1|pid)', subset(trial_frame, condition == 0))
summary(model)


# -- expected score
model <- lmer('expectedScore ~ index * condition + (1|pid)', trial_frame)
summary(model)

model <- lmer('expectedScore ~ index + (1|pid)', subset(trial_frame, condition == 1))
summary(model)
