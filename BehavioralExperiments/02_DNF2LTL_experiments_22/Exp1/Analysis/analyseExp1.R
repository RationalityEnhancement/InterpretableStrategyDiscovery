library(rcompanion)
library(psych)
library(car)
library(ggplot2)
library(scales)
library(effsize)
library(ggpubr)

"
 We have:
  Betweeen factor: condition
  dependent Variable: pa_mean, trialTimeMean, expectedScoreMean, 
"

# IMPORT --------------------------------------------------------------
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
summary(df_finished$totalTime) #duration
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

# test
wilcox.test(pa_mean ~ conditionType, data=df_valid, alternative ='less')



# settings
colors <- c("#3977AF", "#EF8536")
fontsize <- 26
fontsize_title <- 26

# plot violin
p1 <- ggplot(df_valid, aes(x=conditionType, y=pa_mean, fill=conditionType)) +
  geom_violin(alpha = 0.5) +
  scale_fill_manual(values=colors) +
  geom_boxplot(width = 0.1, color = 'black', fill = colors, alpha = 1) + 
  scale_y_continuous(labels = percent, minor_breaks = seq(0 , 0, 0)) +
  theme(legend.position="none") +
  ylab('Mean click agreement') + 
  xlab('') + 
  ggtitle('Click agreement per group')+
  scale_x_discrete(labels=c("flowchart" = "static\ndescriptions", "instructions" = "procedural\ninstructions")) +
  theme(text = element_text(size = fontsize),
        plot.title = element_text(hjust = 0.5, size = fontsize_title), 
        axis.text=element_text(size=fontsize))
p1



# Trial Duration ----------------------------------------------------------
describeBy(df_valid$trialTimeMean, df_valid$conditionType, digits=3, mat=TRUE)

# test
t.test(trialTimeMean ~ conditionType, data=df_valid, var.equal=TRUE, alternative ='greater')


# Expected score ----------------------------------------------------------
describeBy(df_valid$expectedScoreMean, df_valid$conditionType, digits=3, mat=TRUE)

# test
t.test(expectedScoreMean ~ conditionType, data=df_valid, var.equal=TRUE,  alternative ='less')
cohen.d(expectedScoreMean ~ conditionType, data = df_valid)

# correlation
cor.test(df_valid$expectedScoreMean, df_valid$pa_mean)


# plot violin
p2 <- ggplot(df_valid, aes(x=conditionType, y=expectedScoreMean, fill=conditionType)) +
  geom_violin(alpha = 0.5) +
  scale_fill_manual(values=colors) +
  geom_boxplot(width = 0.1, color = 'black', fill = colors, alpha = 1) + 
  scale_y_continuous(limits = c(1, 11.7)) +
  theme(legend.position="none") +
  ylab('Mean expected score') + 
  xlab('') + 
  ggtitle('Expected score per group')+
  scale_x_discrete(labels=c("flowchart" = "static\ndescriptions", "instructions" = "procedural\ninstructions")) +
  theme(text = element_text(size = fontsize),
        plot.title = element_text(hjust = 0.5, size = fontsize_title), 
        axis.text=element_text(size=fontsize))
p2 

pdf('plots/Exp1_results.pdf', width = 14, height = 7)
ggarrange(p1, p2, labels = c("a", "b"), ncol = 2, nrow = 1, font.label = list(size = fontsize))
dev.off()


