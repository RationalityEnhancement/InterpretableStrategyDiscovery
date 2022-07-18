
source('01_format_data.R')

colors <- c("#3977AF", "#EF8536")
fontsize <- 25
fontsize_title <- 25
cond_names <- c('control', 'supported by \n decision aid')
labels <- c('control', 'supported by \n decision aid')

# Mean FSQ per group ------------------------------------------------------

# roadtrip
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
  ggtitle('Far-sightedness in the \n Road Trip task')


# mortgage
# mean FSQ per group
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

# Mean Score per group ------------------------------------------------------

# roadtrip
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
  ggtitle('Performance in the \n Road Trip task')


# mortgage
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



# Mean CA per group ------------------------------------------------------

# roadtrip
df <- aggregate(CA ~ pid + conditionType, subset(tf_valid, task=='roadtrip'), mean)
p5 <- ggplot(df, aes_string(x='conditionType', y='CA', fill='conditionType')) +
  geom_violin(alpha = 0.5) +
  scale_fill_manual(values=colors) +
  geom_boxplot(width = 0.1, color = 'black', fill = colors, alpha = 1) + 
  scale_y_continuous(minor_breaks = seq(0 , 0, 0), labels = percent) +
  scale_x_discrete(labels = cond_names) +
  theme(legend.position="none") +
  theme(text = element_text(size = fontsize),
        plot.title = element_text(hjust = 0.5, size = fontsize_title), 
        axis.text=element_text(size=fontsize)) + 
  ylab('Mean Click Agreement') + 
  xlab('') + 
  ggtitle('Click agreement in the \n Road Trip task')


# mortgage
# mean FSQ per group
df <- aggregate(CA ~ pid + conditionType, subset(tf_valid, task=='mortgage'), mean)
p6 <- ggplot(df, aes_string(x='conditionType', y='CA', fill='conditionType')) +
  geom_violin(alpha = 0.5) +
  scale_fill_manual(values=colors) +
  geom_boxplot(width = 0.1, color = 'black', fill = colors, alpha = 1) + 
  scale_x_discrete(labels = cond_names) +
  scale_y_continuous(minor_breaks = seq(0 , 0, 0), labels = percent) +
  theme(legend.position="none") +
  theme(text = element_text(size = fontsize),
        plot.title = element_text(hjust = 0.5, size = fontsize_title), 
        axis.text=element_text(size=fontsize)) + 
  ylab('Mean Click Agreement') + 
  xlab('') + 
  ggtitle('Click agreement in the \n Mortgage task')

pdf('../plots/Exp2_combined.pdf', width = 14, height = 18)
ggarrange(p2, p1, p6, p5, p4, p3, labels = c("a", "b", "c", "d", "e", "f"), ncol = 2, nrow = 3, font.label = list(size = fontsize))
dev.off()


# FSQ Learning Curves per group -----------------------------------------------------

getLC <- function(df, dv, labels = cond_names, return=FALSE, path=''){
  
  # cast condition as binary variable
  df$condition = df$conditionType == 'instructions'
  df$trial <- df$trial + 1
  
  # aggregate mean
  formula = reformulate(termlabels = 'condition + trial', response = dv)
  data_mean <- aggregate(formula, df, mean)
  
  # aggregate error
  data_N <- aggregate(formula, df, length)
  data_sd <- aggregate(formula, df, sd)
  data_mean$error <- 1.96*data_sd[,c(dv)]/sqrt(data_N[,c(dv)])
  
  #rename
  names(data_mean) <- c('condition', 'trial', 'mean', 'error')
  
  if(return){
    data_mean
    
  } else {
    
    plotLC(data_mean, dv, labels, return, path = path)
  }
}

data <- getLC(subset(tf_valid, task =='roadtrip'), 'FSQ', return = TRUE)
# plot
p1 <- ggplot(data, mapping = aes(x=trial, y=mean, group=condition)) + geom_line(aes(color=condition)) +
  geom_ribbon(subset(data, !condition), mapping=aes(ymin=mean-error, ymax=mean+error), alpha=0.2, fill=colors[1]) +
  geom_ribbon(subset(data, condition), mapping=aes(ymin=mean-error, ymax=mean+error), alpha=0.2, fill=colors[2]) +
  xlab("Trial") +
  ylab('Mean FSQ') +
  ggtitle('Far-sightedness in \nthe Road Trip task') +
  theme_bw() +
  theme(legend.position = "top", plot.title = element_text(hjust = 0.5)) +
  scale_color_manual(values = colors, labels = labels)  +
  theme(text = element_text(size = 15))  + 
  scale_x_continuous(limits = c(1,8), expand = c(0, 0), breaks = seq(1 , 8, 1), minor_breaks = seq(0 , 10, 1))+
  scale_y_continuous(labels = percent, limits=c(0.1, 1)) + 
  theme(text = element_text(size = fontsize),
        plot.title = element_text(hjust = 0.5, size = fontsize_title), 
        axis.text=element_text(size=fontsize),
        legend.title=element_blank(), 
        legend.position="none")
p1



data <- getLC(subset(tf_valid, task =='mortgage'), 'FSQ', return = TRUE)
# plot
p2 <- ggplot(data, mapping = aes(x=trial, y=mean, group=condition)) + geom_line(aes(color=condition)) +
  geom_ribbon(subset(data, !condition), mapping=aes(ymin=mean-error, ymax=mean+error), alpha=0.2, fill=colors[1]) +
  geom_ribbon(subset(data, condition), mapping=aes(ymin=mean-error, ymax=mean+error), alpha=0.2, fill=colors[2]) +
  xlab("Trial") +
  ylab('Mean FSQ') +
  ggtitle('Far-sightedness in \nthe Mortgage task') +  
  theme_bw() +
  theme(legend.position = "top", plot.title = element_text(hjust = 0.5)) +
  scale_color_manual(values = colors, labels = labels)  +
  theme(text = element_text(size = 15))  + 
  scale_x_continuous(limits = c(1,8), expand = c(0, 0), breaks = seq(1 , 8, 1), minor_breaks = seq(0 , 10, 1))+
  scale_y_continuous(labels = percent, limits=c(0.1, 1)) + 
  theme(text = element_text(size = fontsize),
        plot.title = element_text(hjust = 0.5, size = fontsize_title), 
        axis.text=element_text(size=fontsize),
        legend.title=element_blank(), 
        legend.position="none")
p2

# Mean Score Learning Curves -----------------------------------------------------

data <- getLC(subset(tf_valid, task =='roadtrip'), 'score', return = TRUE)
# plot
p3 <- ggplot(data, mapping = aes(x=trial, y=mean, group=condition)) + geom_line(aes(color=condition)) +
  geom_ribbon(subset(data, !condition), mapping=aes(ymin=mean-error, ymax=mean+error), alpha=0.2, fill=colors[1]) +
  geom_ribbon(subset(data, condition), mapping=aes(ymin=mean-error, ymax=mean+error), alpha=0.2, fill=colors[2]) +
  xlab("Trial") +
  ylab('Mean Score') +
  ggtitle('Performance in \nthe Road Trip task') +
  theme_bw() +
  theme(legend.position = "top", plot.title = element_text(hjust = 0.5)) +
  scale_color_manual(values = colors, labels = labels)  +
  theme(text = element_text(size = 15))  + 
  scale_x_continuous(limits = c(1,8), expand = c(0, 0), breaks = seq(1 , 8, 1), minor_breaks = seq(0 , 10, 1))+
  theme(text = element_text(size = fontsize),
        plot.title = element_text(hjust = 0.5, size = fontsize_title), 
        axis.text=element_text(size=fontsize),
        legend.title=element_blank(), 
        legend.position="none")
p3


data <- getLC(subset(tf_valid, task =='mortgage'), 'score', return = TRUE)
# plot
p4 <- ggplot(data, mapping = aes(x=trial, y=mean, group=condition)) + geom_line(aes(color=condition)) +
  geom_ribbon(subset(data, !condition), mapping=aes(ymin=mean-error, ymax=mean+error), alpha=0.2, fill=colors[1]) +
  geom_ribbon(subset(data, condition), mapping=aes(ymin=mean-error, ymax=mean+error), alpha=0.2, fill=colors[2]) +
  xlab("Trial") +
  ylab('Optimal Choices') +
  ggtitle('Performance in \nthe Mortgage task') +
  theme_bw() +
  theme(legend.position = "top", plot.title = element_text(hjust = 0.5)) +
  scale_color_manual(values = colors, labels = labels)  +
  theme(text = element_text(size = 15))  + 
  scale_x_continuous(limits = c(1,8), expand = c(0, 0), breaks = seq(1 , 8, 1), minor_breaks = seq(0 , 10, 1))+
  scale_y_continuous(labels = percent) + 
  theme(text = element_text(size = fontsize),
        plot.title = element_text(hjust = 0.5, size = fontsize_title), 
        axis.text=element_text(size=fontsize),
        legend.title=element_blank(),
        legend.position="none")
p4

pdf('../plots/Exp2_learning_curves.pdf', width = 14, height = 12)
ggarrange(p2, p1, p4, p3, labels = c("a", "b", "c", "d"), ncol = 2, nrow = 2, font.label = list(size = fontsize))
dev.off()
