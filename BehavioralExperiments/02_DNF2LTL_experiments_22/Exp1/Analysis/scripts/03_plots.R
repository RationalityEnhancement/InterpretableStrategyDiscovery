"
 We have:
  Betweeen factor: condition
  dependent Variable: pa_mean, trialTimeMean, expectedScoreMean, 
"

source('01_format_data.R')

# settings
colors <- c("#3977AF", "#EF8536")
labels <- c("static\ndescriptions", "procedural\ninstructions")
fontsize <- 26
fontsize_title <- 26

# Click Agreement - Violin Plot------------------------------------------------------------------

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


# Expected score - Violin Plot----------------------------------------------------------

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

pdf('../plots/Exp1_results.pdf', width = 14, height = 7)
ggarrange(p1, p2, labels = c("a", "b"), ncol = 2, nrow = 1, font.label = list(size = fontsize))
dev.off()


# Click Agreement - Learning curve plot----------------------------------------------------------


getLC <- function(df, dv, labels = c("static descriptions","procedural instructions"), return=FALSE, path=''){
  
  # cast condition as binary variable
  df$condition = (df$condition == 1)
  
  # aggregate mean
  formula = reformulate(termlabels = 'condition + index', response = dv)
  data_mean <- aggregate(formula, df, mean)
  
  # aggregate error
  data_N <- aggregate(formula, df, length)
  data_sd <- aggregate(formula, df, sd)
  data_mean$error <- 1.96*data_sd[,c(dv)]/sqrt(data_N[,c(dv)])
  
  #rename
  names(data_mean) <- c('condition', 'index', 'mean', 'error')
  
  if(return){
    data_mean
    
  } else {
    
    plotLC(data_mean, dv, labels, return, path = path)
  }
}

data <- getLC(trial_frame, 'pa', return = TRUE)

# plot
p3 <- ggplot(data, mapping = aes(x=index, y=mean, group=condition)) + geom_line(aes(color=condition)) +
  geom_ribbon(subset(data, !condition), mapping=aes(ymin=mean-error, ymax=mean+error), alpha=0.2, fill=colors[1]) +
  geom_ribbon(subset(data, condition), mapping=aes(ymin=mean-error, ymax=mean+error), alpha=0.2, fill=colors[2]) +
  xlab("Trial") +
  ylab('Mean click agreement') +
  theme_bw() +
  theme(legend.position = "top", plot.title = element_text(hjust = 0.5)) +
  scale_color_manual(values = colors, labels = labels)  +
  theme(text = element_text(size = 15))  + 
  scale_x_continuous(limits = c(1,10), expand = c(0, 0), breaks = seq(1 , 10, 2), minor_breaks = seq(0 , 10, 1))+
  scale_y_continuous(labels = percent) + 
  theme(text = element_text(size = fontsize),
        plot.title = element_text(hjust = 0.5, size = fontsize_title), 
        axis.text=element_text(size=fontsize),
        legend.title=element_blank())
p3


# Expected Score - Learning curve plot----------------------------------------------------------

data <- getLC(trial_frame, 'expectedScore', return = TRUE)

# plot
p4 <- ggplot(data, mapping = aes(x=index, y=mean, group=condition)) + geom_line(aes(color=condition)) +
  geom_ribbon(subset(data, !condition), mapping=aes(ymin=mean-error, ymax=mean+error), alpha=0.2, fill=colors[1]) +
  geom_ribbon(subset(data, condition), mapping=aes(ymin=mean-error, ymax=mean+error), alpha=0.2, fill=colors[2]) +
  xlab("Trial") +
  ylab('Mean expected score') +
  theme_bw() +
  theme(legend.position = "top", plot.title = element_text(hjust = 0.5)) +
  scale_color_manual(values = colors, labels = labels)  +
  theme(text = element_text(size = 15))  + 
  scale_x_continuous(limits = c(1,10), expand = c(0, 0), breaks = seq(1 , 10, 2), minor_breaks = seq(0 , 10, 1)) +
  theme(text = element_text(size = fontsize),
        plot.title = element_text(hjust = 0.5, size = fontsize_title), 
        axis.text=element_text(size=fontsize), 
        legend.title=element_blank())
p4


pdf('../plots/Exp1_learningcurves.pdf', width = 14, height = 7)
ggarrange(p3, p4, labels = c("a", "b"), ncol = 2, nrow = 1, font.label = list(size = fontsize))
dev.off()







# Click Agreement - Violin Plot - 3 conditions ------------------------------------------------------------------

# settings
colors <- c("#C0C0C0", "#3977AF", "#EF8536")
labels <- c("static", "procedural", "none")
fontsize <- 26
fontsize_title <- 26

# plot violin
p1 <- ggplot(df_combined, aes(x=conditionType, y=pa_mean, fill=conditionType)) +
  geom_violin(alpha = 0.5) +
  scale_fill_manual(values=colors) +
  geom_boxplot(width = 0.1, color = 'black', fill = colors, alpha = 1) + 
  scale_y_continuous(labels = percent, minor_breaks = seq(0 , 0, 0)) +
  theme(legend.position="none") +
  ylab('Mean click agreement') + 
  xlab('') + 
  ggtitle('Click agreement per group')+
  scale_x_discrete("Instructions", labels=c("flowchart" = labels[1], "instructions" = labels[2], 'control'=labels[3])) +
  theme(text = element_text(size = fontsize),
        plot.title = element_text(hjust = 0.5, size = fontsize_title), 
        axis.text=element_text(size=fontsize))
p1


# Expected score - Violin Plot - 3 conditions ----------------------------------------------------------

# plot violin
p2 <- ggplot(df_combined, aes(x=conditionType, y=expectedScoreMean, fill=conditionType)) +
  geom_violin(alpha = 0.5) +
  scale_fill_manual(values=colors) +
  geom_boxplot(width = 0.1, color = 'black', fill = colors, alpha = 1) + 
  scale_y_continuous(limits = c(1, 11.7)) +
  theme(legend.position="none") +
  ylab('Mean expected score') + 
  xlab('') + 
  ggtitle('Expected score per group')+
  scale_x_discrete("Instructions", labels=c("flowchart" = labels[1], "instructions" = labels[2], 'control'=labels[3])) +
  theme(text = element_text(size = fontsize),
        plot.title = element_text(hjust = 0.5, size = fontsize_title), 
        axis.text=element_text(size=fontsize))
p2 

pdf('../plots/Exp1_results_extended.pdf', width = 14, height = 7)
ggarrange(p1, p2, labels = c("a", "b"), ncol = 2, nrow = 1, font.label = list(size = fontsize))
dev.off()


