d = read_csv("optimal_lexicons_for_plot.csv")

d$total_num_words = paste("num words: ", d$total_num_words)
d$total_distal = paste("distal levels: ", d$total_distal)
d$word = as.factor(d$word)


for (m in unique(d$mu)) {
  for (g in unique(d$gamma)) {
    for (l in unique(d$loc)) {
    x = filter(d, gamma == g, mu == m, loc == l)
    title = paste("mu: ", m, "gamma: ", g, "loc: ", l)
     p  =  ggplot(x, aes(x=distal, y=value, fill=word)) + geom_bar(position="stack", stat="identity") + 
        facet_grid(total_num_words ~ total_distal) + 
        xlab("Distal Level") + ylab("value") +
        ggtitle(title)
     print(p)
     ggsave(paste0("optimal_pdfs/", title, ".pdf"))
    }
  }
}
