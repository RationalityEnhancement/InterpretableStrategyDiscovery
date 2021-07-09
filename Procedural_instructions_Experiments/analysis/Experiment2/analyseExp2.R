library(ggpubr)
library(rstatix)
library(scales)
library(ggplot2)
library(psych)
library(WRS2)
library(tidyverse)
library(rcompanion)

# Import ------------------------------------------------------------------
setwd("~/Documents/Dev/Interpretability/InterpretableStrategyDiscovery/Procedural_Instructions_Experiments/analysis/Experiment2")
df <- read.csv('data/dataframe_complete.csv')

# Filter
df_finished <- subset(df , (status != 6 & status != 2))
# Subsets and Factor
df_finished$condition <- factor(df_finished$condition, levels = c(0, 1))

df_valid <- subset(df_finished , attemptsQuiz1 < 4)



# Descriptive------------------------------------------------------------------
nrow(df_finished) - nrow(df_valid)  # excluded
(nrow(df_finished) - nrow(df_valid))/nrow(df_finished) # excluded percentage


sum(grepl('fe', df_finished[ df_finished$gender != '-',]$gender, ignore.case = TRUE)) # female?
summary(strtoi(df_finished$age)) # age
summary(as.numeric(df_finished$totalTime)) #duration
summary(as.numeric(df_finished$attempts)) #attempts
summary(as.numeric(subset(df_finished, condition == 1)$MLcheatTrials)) #attempts
hist(as.numeric(subset(df_finished, condition == 1)$MLcheatTrials))
table(df_valid$conditionType)


getRatio2 <- function(measure, value){
  a1 <- table(df_valid$conditionType, df_valid[,measure] >= value)
  
  print(sum(a1))
  print(a1)
  print(round(prop.table(a1, margin=1),3))
  print(chisq.test(a1))
}


# Analysis RT_PA_MEAN-------------------------------------------------------------
# descriptive
describeBy(df_valid[c("RT_pa_mean")], df_valid$conditionType, digits = 4, mat=TRUE)

# each cell distribution
par(mfrow=c(2,1))
hist(subset(df_valid , condition == 0)$RT_pa_mean, pch = 1, xlab= 'FSP', main='No Training', breaks=seq(0, 1, 0.05), xlim = c(0,1), ylim = c(0,10))
hist(subset(df_valid , condition == 1)$RT_pa_mean, pch = 1, xlab= 'FSP', main='Training', breaks=seq(0, 1, 0.05), xlim = c(0,1), ylim = c(0,10))
par(mfrow=c(1,1))


# normality 
shapiro_test(df_valid$RT_pa_mean)

# Homogneity of variance assumption
fligner.test(RT_pa_mean ~ condition, data=df_valid)

#wilcox
wilcox.test(RT_pa_mean ~ condition, data=df_valid, alternative='less')

# ratios
getRatio2('RT_pa_mean', 0.5)
getRatio2('RT_pa_mean', 0.75)

# plot violine
ggplot(df_valid, aes(x=conditionType, y=RT_pa_mean*100, fill=conditionType)) +
  geom_violin(alpha = 0.5) +
  scale_fill_manual(values=c("#e87442","#6ea7ca")) +
  geom_boxplot(width = 0.1, color = 'black', fill = c("#e87442","#6ea7ca"), alpha = 1) + 
  scale_y_continuous(minor_breaks = seq(0 , 0, 0)) +
  theme(legend.position="none") +
  ylab('FSQ (%)') + 
  xlab('Condition') + 
  scale_x_discrete(labels=c("control" = "No-Training", "Supported Training" = "Training")) +
  ggtitle('FSQ in test trials per group') +
  theme(text = element_text(size = 32), plot.title = element_text(hjust = 0.5, size = 31))


# Analysis RT_PA_MEAN_FT-------------------------------------------------------------
# descriptive
describeBy(df_valid$RT_pa_meanFT, df_valid$conditionType, digits = 4, mat=TRUE)

# PLots
ggplot(df_valid, aes(x=condition, y=RT_pa_meanFT)) + geom_boxplot()
ggplot(aes(y = RT_pa_meanFT, x = conditionType, fill = conditionType), data = df_valid) + geom_boxplot() + xlab("Condition") + ylab("FSP") + ggtitle('FSP over groups')  + theme_bw() + scale_y_continuous(labels = percent, minor_breaks = seq(0 , 0, 0)) + scale_fill_manual(values = c("#EF8536", "#3977AF",  "#4979AF")) + theme(text = element_text(size = 15)) + theme(legend.position = "none")   


# each cell distribution
par(mfrow=c(2,1))
hist(subset(df_valid , condition == 0)$RT_pa_meanFT, pch = 1, xlab= 'FSP', main='No Training', breaks=seq(0, 1, 0.05), xlim = c(0,1), ylim = c(0,30))
hist(subset(df_valid , condition == 1)$RT_pa_meanFT, pch = 1, xlab= 'FSP', main='Training', breaks=seq(0, 1, 0.05), xlim = c(0,1), ylim = c(0,30))
par(mfrow=c(1,1))

# normality 
shapiro_test(df_valid$RT_pa_meanFT)

# Homogneity of variance assumption
fligner.test(RT_pa_meanFT ~ condition, data=df_valid)

# test
wilcox.test(RT_pa_meanFT ~ condition, data=df_valid, alternative='less')

# ratios
getRatio2('RT_pa_meanFT', 0.5)
getRatio2('RT_pa_meanFT', 0.75)





# Correlation train and test FSQ---------------------------------------------------------

#correlation train and test FSQ
df_train <- subset(df_valid, condition == 1)
boxplot(df_train$ML_pa_mean)$stats

# quartiles
a <- subset(df_train, ML_pa_mean >= 0 & ML_pa_mean < 0.64)$RT_pa_mean * 100
b <- subset(df_train, ML_pa_mean >= 0.64 & ML_pa_mean < 0.856)$RT_pa_mean* 100
c <- subset(df_train, ML_pa_mean >= 0.856 & ML_pa_mean < 1)$RT_pa_mean* 100
d <- subset(df_train, ML_pa_mean == 1)$RT_pa_mean* 100

length(a)
length(b)
length(c)
length(d)

par(mar=c(5,6,4,1)+.1)
boxplot(a, b,c ,d, xlab='Training FSQ (%)', ylab='Testing FSQ (%)', names=c("[0,64)","[64,85.6)","[85.6,100)","[100,100]"), col="grey",main ='FSQ in training and testing', cex.main=2, cex.lab=1.9, cex.axis=1.7)  #col="#3977AF"


# boxplot
plotframe <- data.frame(c(a,b,c,d), c( rep("[0,64)", length(a)), rep("[64,85.6)", length(b)), rep("[85.6,100)", length(c)), rep("[100,100]", length(d))))
colnames(plotframe) <- c("values","quartile")
ggplot(plotframe, aes(x=quartile, y = values)) + 
  geom_boxplot() +
  scale_x_discrete(limits=c("[0,64)","[64,85.6)","[85.6,100)","[100,100]")) +
  ggtitle('FSQ in training and testing') + 
  xlab('Training FSQ (%)') + 
  ylab('Testing FSQ (%)') +
  theme(text = element_text(size = 32), plot.title = element_text(hjust = 0.5, size = 31))


# first to third
wilcox.test(a,c)

# third to fourth
wilcox.test(c,d)



# Analyse Learning Effect ---------------------------------------------------------
readArrayColumn <- function(col, n=8){
  res<- data.frame(matrix(ncol=n,nrow=0 ))
  for (el in col) {
    res[nrow(res) + 1,] = scan(text=substr(el, 2, nchar(el)-1), sep=',')
  }
  res
}

trial_comparison = c()

for(trial in 1:8){
  a <- readArrayColumn(subset(df_valid, condition == 0)$RT_pa_complete, 8)[,trial]
  b <- readArrayColumn(subset(df_valid, condition == 1)$RT_pa_complete, 8)[,trial]
  t <- wilcox.test(a, b, alternative='less')
  trial_comparison = append(trial_comparison , t$p.value)
}

# p values for: per trial comparison of FSQ with one sided mann-whitney-u test
print(trial_comparison)




# Analysis RT_Scores-------------------------------------------------------------

# normalize
df_valid$RTscoresMean <- df_valid$RTscoresMean - floor(min(df_valid$RTscoresMean))
df_valid$RTscoresMean<- df_valid$RTscoresMean/max(df_valid$RTscoresMean)

# descriptive
describeBy(df_valid$RTscoresMean, df_valid$conditionType, digits = 4, mat=TRUE)

# PLots
ggplot(aes(y = RTscoresMean, x = conditionType, fill = conditionType), data = df_valid) + geom_boxplot() + xlab("Condition") + ylab("FSP") + ggtitle('Score over groups')  + theme_bw() + scale_y_continuous( minor_breaks = seq(0 , 0, 0)) + scale_fill_manual(values = c("#EF8536", "#3977AF",  "#4979AF")) + theme(text = element_text(size = 15)) + theme(legend.position = "none")   

# normality 
shapiro_test(df_valid$RTscoresMean)

# Homogneity of variance assumption
fligner.test(RTscoresMean ~ condition, data=df_valid)

t.test(RTscoresMean ~ condition, data=df_valid, alternative='greater')



readArrayColumn <- function(col, n=8){
  res<- data.frame(matrix(ncol=n,nrow=0 ))
  for (el in col) {
    res[nrow(res) + 1,] = scan(text=substr(el, 2, nchar(el)-1), sep=',')
  }
  res
}


# Analysis RT_Scores First Half-------------------------------------------------------------

# extract first four trials
a <- readArrayColumn(subset(df_valid, condition == 0)$RTscores, 8)[,0:4]
b <- readArrayColumn(subset(df_valid, condition == 1)$RTscores, 8)[,0:4]

# average per participant
a <- rowMeans(a)
b <- rowMeans(b)

# test
shapiro_test(a)
shapiro_test(b)
t.test(a, b, alternative='greater')


