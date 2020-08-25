library(readxl)
library(tidyverse)

d = read_xlsx("readable_data_tables/europe.xlsx")

#d = fill(d, Language = tidyr::fill(Language)) 

d = d %>% tidyr::fill(Language) %>%
  fill(Type) %>% fill(Place) %>%
  fill(Goal) %>% fill(Source) %>%
  mutate(Place = gsub("^[0-9]", "", Place),
         Source = gsub("^[0-9]", "", Source),
         Goal = gsub("^[0-9]", "", Goal),
         Type = gsub(" [AB]$", "", Type)) %>%
  group_by(Language, Type) %>% 
  mutate(Index=1:n()) %>%
  ungroup() %>%
  filter(Index == 1,
         Type != "SI",
         Type %in% c("D1", "D2", "D3")) # filter to just first of each, no SI

d.long = gather(d, variable, value, -Language, -Index, -Type) %>% 
  group_by(Language) %>%
  mutate(uid = dense_rank(value))

ff = read_csv("~/db/finnish_freqs.csv") %>%
  group_by(word) %>%
  mutate(count = as.numeric(as.character(count))) %>%
  summarise(count=sum(count))
fin = expand.grid(distal=c("D1", "D2", "D3"),
                  type=c("place", "goal", "source"))
words = c("täällä", "siellä", "tuolla",
          "tänne", "sinne", "tuonne",
          "täältä", "sieltä", "tuolta")
fin$word = words
fin = left_join(fin, ff)


ggplot(fin, aes(x=distal, y=count))+ 
  geom_bar(stat="identity") + 
  facet_grid(. ~ type)


fin$Type = as.character(fin$Type)
names(fin) = c("Type", "variable", "finword", "count")
fin$Type = as.character(fin$Type)
fin$variable = as.character(fin$variable)
fin$variable = toupper(fin$variable)

# given number of unique words, scramble them. How do we do?
empty.grid = expand.grid(variable=c("PLACE", "GOAL", "SOURCE"),
                         Type=c("D1", "D2", "D3"),
                         Language = unique(d$Language))
d.long$variable = toupper(d.long$variable)

d.long.full = left_join(empty.grid, d.long) %>%
  arrange(Language, variable, Type) %>%
  group_by(Language, variable) %>%
  fill(value)


d.long.full = left_join(d.long.full, fin)

ent.df = group_by(d.long.full, Language, value) %>%
  summarise(count.sum = sum(count)) %>%
  ungroup() %>%
  group_by(Language) %>%
  mutate(freq = count.sum/sum(count.sum))

ent.df.sum = group_by(ent.df, Language) %>%
  summarise(ent=-sum(freq*log(freq)),
            unique.words = length(unique(value)))

arrange(ent.df.sum, -ent)

arrange(ent.df.sum, ent)

ggplot(ent.df.sum, aes(x=unique.words, y=ent)) + geom_point()
