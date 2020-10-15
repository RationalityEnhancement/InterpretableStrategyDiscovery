library(ggpubr)
library(rstatix)
library(scales)
library(ggplot2)
library(psych)
library(WRS2)
library(tidyverse)
library(rcompanion)

# Import ------------------------------------------------------------------
df <- read.csv('data/Exp4/dataframe_complete.csv')

# Filter
df_valid_all <- subset(df , status != 6)
df_valid <- subset(df_valid_all , cheatTrials < 10)

# Subsets and Factor
df_valid$structure <- factor(df_valid$structure, levels = c("increasing", "decreasing", "constant"))
df_valid$condition <- factor(df_valid$condition, levels = c(0, 1))

df_feed <- subset(df_valid , condition_type == 'actionfeedback')
df_flow <- subset(df_valid , condition_type == 'flowchart')


# Descriptive------------------------------------------------------------------
nrow(df_valid_all) - nrow(df_valid)  # excluded
(nrow(df_valid_all) - nrow(df_valid))/nrow(df_valid_all) # excluded percentage

nrow(df_feed)
nrow(df_flow)

sum(grepl('fe', df_valid_all[ df_valid_all$gender != '-',]$gender, ignore.case = TRUE)) # female?
summary(strtoi(df_valid_all$age)) # age
summary(as.numeric(df_valid_all$totalTime)) #duration
summary(as.numeric(df_valid_all$attempts)) #attempts


# Analysis PA_MEAN-------------------------------------------------------------
# descriptive
describeBy(df_valid[c("pa_mean")], df_valid$condition_type, digits = 4, mat=TRUE)
describeBy(df_valid[c("pa_mean")], list(df_valid$condition_type,df_valid$structure), digits = 4, mat=TRUE)

nrow(df_feed[df_feed$pa_mean > 0.5,])/nrow(df_feed)
nrow(df_flow[df_flow$pa_mean > 0.5,])/nrow(df_flow)

nrow(df_feed[df_feed$pa_mean > 0.8,])/nrow(df_feed)
nrow(df_flow[df_flow$pa_mean > 0.8,])/nrow(df_flow)

# PLots
ggplot(df_valid, aes(x=structure, y=pa_mean)) + geom_boxplot()
ggplot(df_valid, aes(x=condition, y=pa_mean)) + geom_boxplot()

png('plots/Exp4/exp4_ca.png', width = 1800, height=1400, res=300)
ggplot(aes(y = pa_mean, x = structure, fill = condition_type), data = df_valid) + geom_boxplot() + xlab("Environment") + ylab("Click agreement") + ggtitle('Click agreement per tutor and environment') + labs(fill = "Tutor") + theme_bw() + scale_y_continuous(labels = percent, minor_breaks = seq(0 , 0, 0)) + theme(legend.position = "bottom", plot.title = element_text(hjust = 0.5, size = 14), ) + scale_fill_manual(values = c("#EF8536", "#3977AF"), labels = c("Performance feedback", "Flowchart")) + theme(text = element_text(size = 15))   
dev.off()

# each cell distribution
par(mfrow=c(3,2))
hist(subset(df_valid , condition == 0 & structure == 'increasing')$pa_mean, pch = 1, main='inc flowchart')
hist(subset(df_valid , condition == 1 & structure == 'increasing')$pa_mean, pch = 2, main='inc feedback')
hist(subset(df_valid , condition == 0 & structure == 'decreasing')$pa_mean, pch = 3, main='dec flowchart')
hist(subset(df_valid , condition == 1 & structure == 'decreasing')$pa_mean, pch = 4, main='dec feedback')
hist(subset(df_valid , condition == 0 & structure == 'constant')$pa_mean, pch = 5, main='con flowchart')
hist(subset(df_valid , condition == 1 & structure == 'constant')$pa_mean, pch = 6, main='con feedback')
par(mfrow=c(1,1))

# normality 
shapiro_test(df_valid$pa_mean)
ggqqplot(df_valid, "pa_mean", ggtheme = theme_bw()) + facet_grid(structure ~ condition)

# Homogneity of variance assumption
fligner.test(pa_mean ~ interaction(structure,condition), data=df_valid)

# --> robust trimmed anova
t2way(pa_mean ~ structure + condition + structure:condition , data = df_valid)

# plot to resolve interactin effect
interaction.plot(df_valid$structure, df_valid$condition, df_valid$pa_mean)

# post-tests for to resolve interaction effect
wilcox.test(pa_mean ~ condition_type, data=df_valid[df_valid$structure == 'increasing',] )
wilcox.test(pa_mean ~ condition_type, data=df_valid[df_valid$structure == 'constant',] )
wilcox.test(pa_mean ~ condition_type, data=df_valid[df_valid$structure == 'decreasing',] )



# Analysis REW_expected_total-------------------------------------------------------------
# descriptive
describeBy(df_valid[c("rews_exp_total")], df_valid$condition_type, digits = 4, mat=TRUE)
describeBy(df_valid[c("rews_exp_total")], list(df_valid$condition_type,df_valid$structure), digits = 4, mat=TRUE)

# PLots
ggplot(df_valid, aes(x=structure, y=rews_exp_total)) + geom_boxplot()
ggplot(df_valid, aes(x=condition, y=rews_exp_total)) + geom_boxplot()

png('plots/Exp4/exp4_emr.png', width = 1800, height=1400, res=300)
ggplot(aes(y = rews_exp_total/10, x = structure, fill = condition_type), data = df_valid) + geom_boxplot() + xlab("Environment") + ylab("Mean expected score") + ggtitle('Mean expected score per tutor and environment') + labs(fill = "Tutor") + theme_bw() + scale_y_continuous( minor_breaks = seq(0 , 0, 0)) + theme(legend.position = "bottom", plot.title = element_text(hjust = 0.5, size = 14), ) + scale_fill_manual(values = c("#EF8536", "#3977AF"), labels = c("Performance feedback", "Flowchart"))   + theme(text = element_text(size = 15))  
dev.off()

# normality 
shapiro_test(df_valid$rews_exp_total)
ggqqplot(df_valid, "rews_exp_total", ggtheme = theme_bw()) + facet_grid(structure ~ condition)

# Homogneity of variance assumption
fligner.test(rews_exp_total ~ interaction(structure,condition), data=df_valid)

# --> we use robust median trimmed anova
t2way(rews_exp_total ~ structure + condition + structure:condition , data = df_valid)

# plot to resolve interactin effect
interaction.plot(df_valid$structure, df_valid$condition, df_valid$rews_exp_total)

# post-tests for to resolve interaction effect
wilcox.test(rews_exp_total ~ condition_type, data=df_valid[df_valid$structure == 'increasing',] )
wilcox.test(rews_exp_total ~ condition_type, data=df_valid[df_valid$structure == 'constant',] )
wilcox.test(rews_exp_total ~ condition_type, data=df_valid[df_valid$structure == 'decreasing',] )




# Analyse Learning Effect ---------------------------------------------------------
readArrayColumn <- function(col){
  res<- data.frame(matrix(ncol=10,nrow=0 ))
  for (el in col) {
      res[nrow(res) + 1,] = scan(text=substr(el, 2, nchar(el)-1), sep=',')
  }
  res
}

groupwiseMedianWrapper <- function(v){
  groupwiseMedian(data = data.frame(v), conf = 0.95, var = 'v', bca = FALSE, percentile = TRUE, R = 10000)
}

getLC <- function(v, structure, condition){
  stat <- apply(readArrayColumn(v), 2, groupwiseMedianWrapper)
  
  med <- c()
  plow <- c()
  pup <- c()
  
  for (el in stat) {
    med  <- append(med,  el$Median)
    plow <- append(plow, el$Percentile.lower)
    pup  <- append(pup,  el$Percentile.upper)
  }
  
  lc <- data.frame(med, c(1:10), plow, pup, structure, condition)
  colnames(lc) <- c("ya", "xa", 'plow', 'pup' , 'structure', 'condition')
  lc$structure <- factor(lc$structure, levels = c("increasing", "decreasing", "constant"))
  lc$condition <- factor(lc$condition, levels = c(0, 1))
   
  lc
}

getLCDF <- function(col){
  res <- data.frame(readArrayColumn(col), df_valid$condition, df_valid$structure)
  colnames(res)[11] <- "condition"
  colnames(res)[12] <- "structure"
  res
}


plotto_env <- function(env, dv, perc=FALSE){
  df_lc_all <- subset(df_lc_all, structure ==env)
  sign_vector <- which(c(wilcox.test(X1 ~ condition, data=df_lc_all)$p.value < 0.05, 
                         wilcox.test(X2 ~ condition, data=df_lc_all)$p.value < 0.05, 
                         wilcox.test(X3 ~ condition, data=df_lc_all)$p.value < 0.05, 
                         wilcox.test(X4 ~ condition, data=df_lc_all)$p.value < 0.05, 
                         wilcox.test(X5 ~ condition, data=df_lc_all)$p.value < 0.05, 
                         wilcox.test(X6 ~ condition, data=df_lc_all)$p.value < 0.05, 
                         wilcox.test(X7 ~ condition, data=df_lc_all)$p.value < 0.05, 
                         wilcox.test(X8 ~ condition, data=df_lc_all)$p.value < 0.05, 
                         wilcox.test(X9 ~ condition, data=df_lc_all)$p.value < 0.05, 
                         wilcox.test(X10 ~ condition, data=df_lc_all)$p.value < 0.05))
  
  pl <- ggplot(subset(df_lc, structure ==env), mapping = aes(x=xa, y=ya, group=condition)) + geom_line(aes(color=condition))
  pl <- pl  + geom_ribbon(subset(df_lc, structure ==env & condition == 0), mapping=aes(ymin=plow, ymax=pup), alpha=0.2, fill="#3977AF")
  pl <- pl  + geom_ribbon(subset(df_lc, structure ==env & condition == 1), mapping=aes(ymin=plow, ymax=pup), alpha=0.2, fill="#EF8536")
  pl <- pl + annotate("text", x = sign_vector, y = mean(c(max(subset(df_lc,structure ==env)$pup),max((subset(df_lc,structure ==env)$ya)))), label = "*", size = 6)
  
  pl <- pl + xlab("Trials") + ylab(dv) + ggtitle(paste(dv, sprintf('over time per tutor \n for the %s environment', env))) + theme_bw() + scale_x_continuous(minor_breaks = seq(0 , 0, 0), breaks = c(0,1,2,3,4,5,6,7,8,9,10)) + theme(legend.position = "bottom", plot.title = element_text(hjust = 0.5))
  pl <- pl  + labs(color="Tutor")+ scale_color_manual(values = c("#3977AF", "#EF8536"), labels = c("Flowchart","Performance feedback"))   + theme(text = element_text(size = 15))  
  
  if(perc == TRUE){
    pl <- pl + scale_y_continuous(labels = percent)
  }
  
  pl
  
}

plotto_cond <- function(dv, perc=FALSE){
  
  sign_vector <- which(c(wilcox.test(X1 ~ condition, data=df_lc_all)$p.value < 0.05, 
                   wilcox.test(X2 ~ condition, data=df_lc_all)$p.value < 0.05, 
                   wilcox.test(X3 ~ condition, data=df_lc_all)$p.value < 0.05, 
                   wilcox.test(X4 ~ condition, data=df_lc_all)$p.value < 0.05, 
                   wilcox.test(X5 ~ condition, data=df_lc_all)$p.value < 0.05, 
                   wilcox.test(X6 ~ condition, data=df_lc_all)$p.value < 0.05, 
                   wilcox.test(X7 ~ condition, data=df_lc_all)$p.value < 0.05, 
                   wilcox.test(X8 ~ condition, data=df_lc_all)$p.value < 0.05, 
                   wilcox.test(X9 ~ condition, data=df_lc_all)$p.value < 0.05, 
                   wilcox.test(X10 ~ condition, data=df_lc_all)$p.value < 0.05))
  
  pl <- ggplot(df_lc, mapping = aes(x=xa, y=ya, group=condition)) + geom_line(aes(color=condition))
  pl <- pl  + geom_ribbon(subset(df_lc, condition == 0), mapping=aes(ymin=plow, ymax=pup), alpha=0.2, fill="#3977AF")
  pl <- pl  + geom_ribbon(subset(df_lc, condition == 1), mapping=aes(ymin=plow, ymax=pup), alpha=0.2, fill="#EF8536")
  pl <- pl + annotate("text", x = sign_vector, y = mean(c(max(df_lc$pup),max(df_lc$ya))), label = "*", size = 6)
  
  pl <- pl + xlab("Trials") + ylab(dv) + ggtitle(paste(dv,'over time per tutor')) + theme_bw() + scale_x_continuous(minor_breaks = seq(0 , 0, 0), breaks = c(0,1,2,3,4,5,6,7,8,9,10)) + theme(legend.position = "bottom", plot.title = element_text(hjust = 0.5))
  pl <- pl  + labs(color="Tutor")+ scale_color_manual(values = c("#3977AF", "#EF8536"), labels = c("Flowchart","Performance feedback"))   + theme(text = element_text(size = 15))  
  
  if(perc == TRUE){
    pl <- pl + scale_y_continuous(labels = percent)
  }
  
  pl
  
}


#-------  rews_exp plotto1 
df_lc_all <- getLCDF(df_valid$rews_exp)

df_lc <- rbind( getLC(subset(df_flow , structure == 'increasing')$rews_exp, 'increasing', 0),
                getLC(subset(df_flow , structure == 'decreasing')$rews_exp,'decreasing', 0),
                getLC(subset(df_flow , structure == 'constant')$rews_exp, 'constant', 0),
                getLC(subset(df_feed , structure == 'increasing')$rews_exp, 'increasing', 1),
                getLC(subset(df_feed , structure == 'decreasing')$rews_exp, 'decreasing', 1),
                getLC(subset(df_feed , structure == 'constant')$rews_exp, 'constant', 1))


png('plots/Exp4/exp4_emr_time_increasing.png', width = 1800, height=1400, res=300)
plotto_env('increasing', "Median expected score")
dev.off()

png('plots/Exp4/exp4_emr_time_decreasing.png', width = 1800, height=1400, res=300)
plotto_env('decreasing', "Median expected score")
dev.off()

png('plots/Exp4/exp4_emr_time_constant.png', width = 1800, height=1400, res=300)
plotto_env('constant', "Median expected score")
dev.off()


#-------  rews_exp plotto2

df_lc_all <- getLCDF(df_valid$rews_exp)

df_lc <- rbind( getLC(df_flow$rews_exp, 'mixed', 0),
                getLC(df_feed$rews_exp, 'mixed', 1))


png('plots/Exp4/exp4_emr_time_tutor.png', width = 1800, height=1400, res=300)
plotto_cond("Median expected score")
dev.off()



#-------------------  pa_complete

df_lc_all <- getLCDF(df_valid$pa_complete)

df_lc <- rbind( getLC(subset(df_flow , structure == 'increasing')$pa_complete, 'increasing', 0),
                getLC(subset(df_flow , structure == 'decreasing')$pa_complete,'decreasing', 0),
                getLC(subset(df_flow , structure == 'constant')$pa_complete, 'constant', 0),
                getLC(subset(df_feed , structure == 'increasing')$pa_complete, 'increasing', 1),
                getLC(subset(df_feed , structure == 'decreasing')$pa_complete, 'decreasing', 1),
                getLC(subset(df_feed , structure == 'constant')$pa_complete, 'constant', 1))

png('plots/Exp4/exp4_ca_time_increasing.png', width = 1800, height=1400, res=300)
plotto_env('increasing', "Median click agreement", perc = TRUE)
dev.off()

png('plots/Exp4/exp4_ca_time_decreasing.png', width = 1800, height=1400, res=300)
plotto_env('decreasing', "Median click agreement", perc = TRUE)
dev.off()

png('plots/Exp4/exp4_ca_time_constant.png', width = 1800, height=1400, res=300)
plotto_env('constant', "Median click agreement", perc = TRUE)
dev.off()


#-------  pa_complete plotto2

df_lc_all <- getLCDF(df_valid$pa_complete)

df_lc <- rbind( getLC(df_flow$pa_complete, 'mixed', 0),
                getLC(df_feed$pa_complete, 'mixed', 1))


png('plots/Exp4/exp4_ca_time_tutor.png', width = 1800, height=1400, res=300)
plotto_cond("Median click agreement", perc = TRUE)
dev.off()

