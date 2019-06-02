library(ggplot2)
data=read.table("prestige.txt", head=TRUE, sep="\t", quote="")
data$prestige=as.factor(data$prestige)
ggplot(data, aes(x=prestige, y=100*val)) + geom_violin() + xlab("") + ylab("Event ratio") + geom_dotplot(binaxis='y', stackdir='center', dotsize=0.3) + stat_summary(fun.y=median, geom="point", size=2, color="red") + theme(axis.text.x=element_text(size=14), axis.title.y=element_text(size=14, margin = margin(t = 0, r = 12, b = 0, l = 0))) 