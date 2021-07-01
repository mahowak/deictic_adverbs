# optimal lexicons to tables
# this script reads the file mi_test_1.csv and converts the entries into those readable by the script count_theta_patterns.py
library(tidyverse)
library(tidyr)

df_original <- read.csv("mi_test_1.csv") 

k <- 1
for (i in 1:nrow(df_original)){
  if (df_original$Language[i]=='simulated'){
    df_original$Language[i] = paste0('simulated', k)
    k <- k + 1
  }
}

df <- df_original %>% pivot_longer(
  cols = paradigm_names(df_original), 
  names_to = "Type",
  values_to = "Word") %>%
  mutate(Word = letters[Word + 1]) %>%
  separate(col = "Type",
           into = c("r", "theta"),
           sep = '_') %>%
  pivot_wider(names_from = 'theta',
              values_from = 'Word') %>%
  select(c('Language', 'r', 'place', 'goal', 'source'))

colnames(df) <- c('Language', 'Type', 'Place', 'Goal', 'Source')
 
write.csv(file = "lexicon_table.csv", df, row.names = FALSE)

paradigm_names <- function(df){
  names <- colnames(df)
  return(names[grepl("^D", colnames(df))])
}


