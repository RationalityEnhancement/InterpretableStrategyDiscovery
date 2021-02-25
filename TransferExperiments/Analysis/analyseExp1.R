library(ggpubr)
library(rstatix)
library(scales)
library(ggplot2)
library(psych)
library(WRS2)
library(tidyverse)
library(rcompanion)

# Import ------------------------------------------------------------------
setwd("~/Documents/Dev/Interpretability/FinalCgSci21/Analysis")
df <- read.csv('data/Experiment1/dataframe_complete.csv')

#df <- subset(df, condition == 1)

# Filter
df_valid_all <- subset(df , (status != 6 & status != 2))
# Subsets and Factor
df_valid_all$condition <- factor(df_valid_all$condition, levels = c(0, 1))

df_valid <- subset(df_valid_all , attemptsQuiz1 < 4)




# Descriptive------------------------------------------------------------------
nrow(df_valid_all) - nrow(df_valid)  # excluded
(nrow(df_valid_all) - nrow(df_valid))/nrow(df_valid_all) # excluded percentage


sum(grepl('fe', df_valid_all[ df_valid_all$gender != '-',]$gender, ignore.case = TRUE)) # female?
summary(strtoi(df_valid_all$age)) # age
summary(as.numeric(df_valid_all$totalTime)) #duration
summary(as.numeric(df_valid_all$attempts)) #attempts
summary(as.numeric(subset(df_valid_all, condition == 1)$MLcheatTrials)) #attempts
hist(as.numeric(subset(df_valid_all, condition == 1)$MLcheatTrials))
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

# PLots
ggplot(df_valid, aes(x=condition, y=RT_pa_mean)) + geom_boxplot()

ggplot(aes(y = RT_pa_mean, x = conditionType, fill = conditionType), data = df_valid) + geom_boxplot() + xlab("Condition") + ylab("FSP") + ggtitle('FSP over groups')  + theme_bw() + scale_y_continuous(labels = percent, minor_breaks = seq(0 , 0, 0)) + scale_fill_manual(values = c("#EF8536", "#3977AF",  "#4979AF")) + theme(text = element_text(size = 15)) + theme(legend.position = "none")   

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


getRatio2('RT_pa_mean', 0.5)
getRatio2('RT_pa_mean', 0.75)


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


getRatio2('RT_pa_meanFT', 0.5)
getRatio2('RT_pa_meanFT', 0.75)





# Correlation ---------------------------------------------------------

#correlation train and test FSP
df_train <- subset(df_valid, condition == 1)
boxplot(df_train$ML_pa_mean)$stats

a <- subset(df_train, ML_pa_mean >= 0 & ML_pa_mean < 0.64)
b <- subset(df_train, ML_pa_mean >= 0.64 & ML_pa_mean < 0.856)$RT_pa_mean * 100
c <- subset(df_train, ML_pa_mean >= 0.856 & ML_pa_mean < 1)
d <- subset(df_train, ML_pa_mean == 1)

length(a)
length(b)
length(c)
length(d)

par(mar=c(5,6,4,1)+.1)
boxplot(a, b,c ,d, xlab='Training FSQ (%)', ylab='Testing FSQ (%)', names=c("[0,64)","[64,85.6)","[85.6,100)","[100,100]"), col="grey",main ='FSQ in training and testing', cex.main=2, cex.lab=1.8, cex.axis=1.6)  #col="#3977AF"

wilcox.test(b,d)
cor.test(df_train$ML_pa_mean, df_train$RT_pa_mean, method='spearman')



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




# Analysis RT_Scores-------------------------------------------------------------
# descriptive
df_valid$RTscoresMean <- df_valid$RTscoresMean - floor(min(df_valid$RTscoresMean))
df_valid$RTscoresMean<- df_valid$RTscoresMean/max(df_valid$RTscoresMean)

describeBy(df_valid$RTscoresMean, df_valid$conditionType, digits = 4, mat=TRUE)

# PLots
ggplot(df_valid, aes(x=condition, y=RTscoresExpMean)) + geom_boxplot()

ggplot(aes(y = RTscoresMean, x = conditionType, fill = conditionType), data = df_valid) + geom_boxplot() + xlab("Condition") + ylab("FSP") + ggtitle('Score over groups')  + theme_bw() + scale_y_continuous( minor_breaks = seq(0 , 0, 0)) + scale_fill_manual(values = c("#EF8536", "#3977AF",  "#4979AF")) + theme(text = element_text(size = 15)) + theme(legend.position = "none")   


# each cell distribution
par(mfrow=c(2,1))
hist(subset(df_valid , condition == 0)$RTscoresMean, pch = 1, xlab= 'FSP', main='No Training', breaks=seq(200, 450, 10), xlim = c(200,450), ylim = c(0,10))
hist(subset(df_valid , condition == 1)$RTscoresMean, pch = 1, xlab= 'FSP', main='Training', breaks=seq(200, 450, 10),, xlim = c(200,450), ylim = c(0,10))
par(mfrow=c(1,1))

# normality 
shapiro_test(df_valid$RTscoresMean)

# Homogneity of variance assumption
fligner.test(RTscoresMean ~ condition, data=df_valid)

t.test(RTscoresMean ~ condition, data=df_valid, alternative='greater')



readArrayColumn <- function(col, n=8){
  res<- data.frame(matrix(ncol=n,nrow=0 ))
  for (el in col) {
    el <- substr(el, 2, nchar(el)-1)
    #print(el)
    res[nrow(res) + 1,] = scan(text=substr(el, 2, nchar(el)-1), sep=',')
  }
  res
}

a <- readArrayColumn(subset(df_valid, condition == 0)$RTscores, 8)[,0:4]
b <- readArrayColumn(subset(df_valid, condition == 1)$RTscores, 8)[,0:4]
a <- rowMeans(a)
b <- rowMeans(b)

hist(a, breaks = 20)
hist(b, breaks = 20)
shapiro_test(a)
shapiro_test(a)
t.test(a, b, alternative='greater')


