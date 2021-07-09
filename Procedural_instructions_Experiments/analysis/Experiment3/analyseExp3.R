library(ggpubr)
library(rstatix)
library(scales)
library(ggplot2)
library(psych)
library(WRS2)
library(tidyverse)
library(rcompanion)
library(vcd)
library(flextable)


# Import ------------------------------------------------------------------
setwd("~/Documents/Dev/Interpretability/InterpretableStrategyDiscovery/Procedural_Instructions_Experiments/analysis/Experiment3")
df_complete <- read.csv('data/dataframe_complete.csv')

t <- read.csv('data/dataframe_mortgage_final.csv', sep=";")
t <- t[order(t$pid),]
df_complete$mortgageChoice  <- t$mortgageChoice


# choose discounting function
if(TRUE){
        df_complete$PB3 <- df_complete$PB3_exp
        df_complete$PB6 <- df_complete$PB6_exp
        df_complete$idf0_3 <- df_complete$idf0_3_exp
        df_complete$idf0_6 <- df_complete$idf0_6_exp
        df_complete$idf6_9 <- df_complete$idf6_9_exp
        df_complete$idf6_12 <- df_complete$idf6_12_exp 
} else {
        df_complete$PB3 <- df_complete$PB3_hyp
        df_complete$PB6 <- df_complete$PB6_hyp 
        df_complete$idf0_3 <- df_complete$idf0_3_hyp 
        df_complete$idf0_6 <- df_complete$idf0_6_hyp
        df_complete$idf6_9 <- df_complete$idf6_9_hyp 
        df_complete$idf6_12 <- df_complete$idf6_12_hyp 
}


# Filter
df_finished <- subset(df_complete  , (status != 6 & status != 2))
df_finished$PB_combined <- (as.numeric(df_finished$PB3) + as.numeric(df_finished$PB6) + (1-df_finished$mortgageChoice))
df_finished$FS_combined_bin <- (as.numeric(df_finished$PB3) + as.numeric(df_finished$PB6) + (1-df_finished$mortgageChoice)) == 0
df_finished$idf_mean <- rowMeans(data.frame(df_finished$idf6_12,df_finished$idf6_9,df_finished$idf0_6,df_finished$idf0_3))

df_valid <- subset(df_finished , sanityFail == 0 & mortgageChoice != -1)

# Subsets and Factor
df_valid$condition <- factor(df_valid$condition, levels = c(0, 1))


# Descriptive------------------------------------------------------------------
nrow(df_finished) - nrow(df_valid)  # excluded
(nrow(df_finished) - nrow(df_valid))/nrow(df_finished) # excluded percentage


sum(grepl('fe', df_finished[ df_finished$gender != '-',]$gender, ignore.case = TRUE)) # female?
summary(strtoi(df_finished$age)) # age
summary(as.numeric(df_finished$totalTime)) #duration
table(df_valid$conditionType)
table(df_valid$MLcheatTrials)
table(df_valid$MLclicksNumber)


# Analysis Present Bias t = 3months-------------------------------------------------------------

# descriptive
dt <- table(df_valid$conditionType, df_valid$PB3)
dt.prop <- prop.table(dt, margin=1)

dt
round(dt.prop, 3)

# test
chisq.test(dt)
sqrt(chisq.test(dt)$statistic / nrow(df_valid))

# each cell distribution
sd= sqrt(dt.prop[,1]*dt.prop[,2])
err = sd/sqrt(rowSums(dt))
err = 1.96*err

# store plot
pd <- data.frame('conditionType'=  factor(c("No-Training","Training"), levels = c("No-Training","Training")), 'value'=as.vector(dt.prop[,1]), 'err' = as.vector(err), 'task'= rep('ITC k=3', 2))



# Analysis Present Bias t = 6 months-------------------------------------------------------------
# descriptive
dt <- table(df_valid$conditionType, df_valid$PB6)
dt.prop <- prop.table(dt, margin=1)

dt
round(dt.prop, 3)
chisq.test(dt)
sqrt(chisq.test(dt)$statistic / nrow(df_valid))

# each cell distribution
sd= sqrt(dt.prop[,1]*dt.prop[,2])
err = sd/sqrt(rowSums(dt))
err = 1.96*err

pd <- rbind(pd, data.frame('conditionType'=  factor(c("No-Training","Training"), levels = c("No-Training","Training")), 'value'=as.vector(dt.prop[,1]), 'err' = as.vector(err), 'task'= rep('ITC k=6', 2)))



# Analysis Mortgage Choice-------------------------------------------------------------
# descriptive
dt <- table(df_valid$conditionType, df_valid$mortgageChoice)
dt.prop <- prop.table(dt, margin=1)
dt
round(dt.prop, 3)
chisq.test(dt)
sqrt(chisq.test(dt)$statistic / nrow(df_valid))

# each cell distribution
sd= sqrt(dt.prop[,1]*dt.prop[,2])
err = sd/sqrt(rowSums(dt))
err = 1.96*err


pd <- rbind(pd, data.frame('conditionType'=  factor(c("No-Training","Training"), levels = c("No-Training","Training")), 'value'=as.vector(dt.prop[,2]), 'err' = as.vector(err), 'task'=rep('Mortgage Choice', 2)))



# Analysis Combined PB-------------------------------------------------------------
# descriptive
dt <- table(df_valid$conditionType, df_valid$FS_combined_bin)
dt.prop <- prop.table(dt, margin=1)
dt
round(dt.prop, 3)
chisq.test(dt)
sqrt(chisq.test(dt)$statistic / nrow(df_valid))

# each cell distribution
sd= sqrt(dt.prop[,1]*dt.prop[,2])
err = sd/sqrt(rowSums(dt))
err = 1.96*err


# combined plot
pd <- rbind(pd, data.frame('conditionType'=  factor(c("No-Training","Training"), levels = c("No-Training","Training")), 'value'=as.vector(dt.prop[,2]), 'err' = as.vector(err), 'task'=rep('Composite', 2)))


par(mar=c(5,6,4,1)-1)
ggplot(aes(y = value, x = task, fill=conditionType), data = pd)  +
        geom_bar(stat="identity", position=position_dodge(), width = 0.75) + 
        scale_y_continuous(labels = percent, minor_breaks = seq(0 , 0, 0)) + 
        scale_x_discrete(guide = guide_axis(n.dodge=2), limits=c("ITC k=3","ITC k=6","Mortgage Choice","Composite")) +
        geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.1, position=position_dodge(.75)) + 
        theme(text = element_text(size = 35), plot.title = element_text(size=35, hjust = 0.5))  + 
        scale_fill_manual(values=c("#e87442","#6ea7ca")) +
        xlab("Task") + 
        ylab("Group proportion") + 
        ggtitle('Far-sighted beahvior per task and group')  + 
        labs(fill = "") +
        theme(legend.position="top", panel.grid.major.x = element_blank())




# idf mean

ggplot(aes(y = idf_mean, x =conditionType, fill = conditionType), data = df_valid) + geom_boxplot() + xlab("Group") + ylab("Discount Factor") + ggtitle('Discount factor per group')  + theme(plot.title = element_text(hjust = 0.5))  + scale_fill_manual(values = c("#EF8536", "#3977AF",  "#3977AF")) + theme(text = element_text(size = 15)) + theme(legend.position = "none") + scale_x_discrete(labels = c('No-Training', 'Training')) #+ ylim(0, 0.5) 


describeBy(df_valid$idf_mean, df_valid$condition)

# each cell distribution
par(mfrow=c(2,1))
hist(subset(df_valid , condition == 0)$idf_mean, pch = 1, xlab= 'Present Bias Strength', main='No Training', breaks=seq(0, 1.3, 0.01))
hist(subset(df_valid , condition == 1)$idf_mean, pch = 1, xlab= 'Present Bias Strength', main='Training',  breaks=seq(0, 1.3, 0.01))
par(mfrow=c(1,1))

# normality 
shapiro_test(df_valid$idf_mean)

# Homogneity of variance assumption
fligner.test(idf_mean ~ condition, data=df_valid)

#wilcox
wilcox.test(idf_mean ~ condition, data=df_valid)s

# Analysis Present idfs-------------------------------------------------------------

pd2 <- data.frame('idf'= df_valid$idf0_3, 'Group'=df_valid$conditionType, 'idf_type' = 'idf0_3')
pd2 <- rbind(pd2, data.frame('idf'= df_valid$idf0_6, 'Group'= df_valid$conditionType, 'idf_type' = 'idf0_6'))
pd2 <- rbind(pd2, data.frame('idf'= df_valid$idf6_9, 'Group'= df_valid$conditionType, 'idf_type' = 'idf6_9'))
pd2 <- rbind(pd2, data.frame('idf'= df_valid$idf6_12, 'Group'= df_valid$conditionType, 'idf_type' = 'idf6_12'))
pd2 <- rbind(pd2, data.frame('idf'= df_valid$idf_mean,  'Group'= df_valid$conditionType, 'idf_type' = 'mean(idf)'))
pd2[pd2$Group == 'no-training',]$Group <- 'No-Training'
        
ggplot(aes(y = idf, x = idf_type, fill = Group), data = pd2) + 
        geom_boxplot() +
        scale_fill_manual(values = c("#e87442","#6ea7ca"))  +
        scale_x_discrete(labels = c('t=0/k=3', 't=0/k=6', 't=6/k=3', 't=6/k=6', 'Mean')) + 
        xlab("Parameter") +
        ylab("Discount Factor") +
        ggtitle('Discount factors per group') +
        labs(fill = "") +
        theme(legend.position="top") +
        theme(text = element_text(size = 36), plot.title = element_text(size=36, hjust = 0.5))




# Correlation ---------------------------------------------------------

# -> subset
df_train <- subset(df_valid, condition != 0)
hist(df_train$ML_pa_mean)

boxplot(df_train$ML_pa_mean)$stats

# test
a <- 3- subset(df_train, ML_pa_mean >= 0 & ML_pa_mean < 0.025)$PB_combined
b <- 3 - subset(df_train, ML_pa_mean >= 0.025 & ML_pa_mean < 0.529)$PB_combined
c <- 3 -subset(df_train, ML_pa_mean >= 0.529 & ML_pa_mean < 0.897)$PB_combined
d <- 3 -subset(df_train, ML_pa_mean >= 0.897)$PB_combined

length(a)
length(b)
length(c)
length(d)

boxplot(a, b, c ,d, xlab='Training FSQ (%)', ylab='Far-sighted Decisions', names=c("[0,2.5)","[2.5,52.9)","[52.9,89.7)","[89.7,100]"), col="grey",main ='FSQ influence on far-sightedness') #col="#3977AF"

wilcox.test(c,d)



pd3 <- data.frame(Group = "[0,2.5)", value = mean(a), err = 1.96*(sd(a)/sqrt(length(a)) ))
pd3 <- rbind(pd3, data.frame(Group = "[2.5,52.9)", value = mean(b), err = 1.96*(sd(a)/sqrt(length(a)))  ))               
pd3 <- rbind(pd3, data.frame(Group = "[52.9,89.7)", value = mean(c), err = 1.96*(sd(a)/sqrt(length(a))) ))        
pd3 <- rbind(pd3, data.frame(Group = "[89.7,100]", value = mean(d), err = 1.96*(sd(a)/sqrt(length(a))) ))

p <- ggplot(pd3) + geom_bar( aes(x=Group, y=value), stat="identity", fill="darkgrey") + geom_errorbar( aes(x=Group, ymin=value-err, ymax=value+err), width=0.2, size=0.4)
p <- p + ggtitle("FSQ influence on far-sightedness") + ylab("Far-sighted Decisions") + xlab("Training FSQ (%)") + ylim(0, 3)
p + theme(plot.title = element_text(hjust = 0.5), text = element_text(size=26), axis.title.x= element_text(vjust = -0.1))




