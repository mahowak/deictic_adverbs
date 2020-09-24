library(tidyverse)
library(RColorBrewer)

d = read_csv("optimal_lexicons_for_plot.csv")

d$total_num_words = paste("num words: ", d$total_num_words)
d$total_distal = paste("distal levels: ", d$total_distal)
#d$gamma = paste("gamma: ", d$gamma)
d$word = as.factor(d$word)


for (m in unique(d$mu)) {
    x = filter(d, mu == m)
    title = paste("mu: ", m) 
     p  =  ggplot(x, aes(x=distal, y=value, fill=word)) + geom_bar(position="stack", stat="identity") + 
        facet_grid(total_num_words + gamma ~ total_distal + loc) + 
        xlab("Distal Level") + ylab("value") +
        ggtitle(title) +
       theme_bw(12) + 
       scale_fill_brewer(palette="Set1")
  
     print(p)
     ggsave(paste0("optimal_pdfs/", title, ".pdf"), width=15, height=15)
}


############## make categorical
e = group_by(d, mu, gamma, loc, total_distal, total_num_words, distal) %>%
  mutate(m=max(value)) %>%
  filter(value == m) %>%
  mutate(value = 1)

for (m in unique(d$mu)) {
  x = filter(d, mu == m)
  title = paste("mu: ", m) 
  p  =  ggplot(x, aes(x=distal, y=loc, fill=word)) + geom_tile()
    facet_grid(total_num_words + gamma ~ total_distal + loc) + 
    xlab("Distal Level") + ylab("value") +
    ggtitle(title) +
    theme_bw(12) + 
    scale_fill_brewer(palette="Set1")
  
  print(p)
  ggsave(paste0("optimal_pdfs/det", title, ".pdf"), width=15, height=15)
}
