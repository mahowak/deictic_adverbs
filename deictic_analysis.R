library(readxl)
library(tidyverse)
library(ggrepel)

eur = read_xlsx("readable_data_tables/europe.xlsx") %>% mutate(area="europe")
afr = read_xlsx("readable_data_tables/africa.xlsx") %>% mutate(area="africa")
asia = read_xlsx("readable_data_tables/asia.xlsx") %>% mutate(area="asia")
d = bind_rows(eur, afr, asia)
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

d.long = gather(d, variable, value, -Language, -Index, -Type, -area) %>% 
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
d.long.full$uid = d.long.full$uid - 1

d.long.full = mutate(d.long.full, deictic_var = case_when(variable == "PLACE" ~ 0,
                                    variable == "GOAL" ~ 3,
                                    variable == "SOURCE" ~ 6),
                     deictic_type = case_when(Type == "D1" ~ 0,
                                     Type == "D2" ~ 1,
                                     Type == "D3" ~ 2),
                     deictic_index = deictic_var + deictic_type)
d.long.full$deictic_index = with(d.long.full, deictic_var + deictic_type)
d.long.full = group_by(d.long.full, Language) %>%
  mutate(uid = dense_rank(value) - 1)
write_csv(select(ungroup(d.long.full), Language, uid, deictic_index, area), "processed_datasheets/all.csv")


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


############ p(u|m)

get.p.u.m = function(place, distal, mu) {
  d = expand.grid(distal=c(1,2,3),
                  place=c(1,2,3))
  d$current = 1 * .1^(abs(d$distal - distal) + abs(d$place - place))
  print(d)
  
  d.new = mutate(d, distal = case_when(distal == 1 ~ "D1",
                               distal == 2 ~ "D2",
                               distal == 3 ~ "D3"),
         place = case_when(place == 1 ~ "goal",
                           place == 2 ~ "location",
                           place ==3 ~ "source")) 
  d.new$current = d.new$current/sum(d.new$current)
    p = ggplot(data=d.new, aes(x=distal, y=current)) +
    geom_bar(stat="identity") +
    facet_grid(. ~ place) + ggtitle(paste(d.new[d.new$current == max(d.new$current), c("distal", "place")][1],
                                          d.new[d.new$current == max(d.new$current), c("distal", "place")][2]))
  print(p)
}
d = get.p.u.m(1, 1)


########
d.long.full$a = paste(d.long.full$variable, d.long.full$Type)
d2 = ungroup(d.long.full) %>% select(a, Language, value) %>%
  spread(a, value)
filter(d2, `GOAL D2` == `GOAL D1`)
filter(d2, `PLACE D2` == `PLACE D1`)
filter(d2, `SOURCE D2` == `SOURCE D1`)


###### analyze frontier
library(tidyverse)
d = read_csv("mi_test_1.csv")

d$Language = substr(d$Language, 1, 9)
d$IsSim = d$Language == "simulated"
ggplot(filter(d, Language == "simulated"), aes(x=`I[U;W]`, y=`I[M;W]`)) + geom_point( colour="gray", alpha=.1) +
  geom_jitter(data=filter(d, Language != "simulated", Language != "optimal"), aes(x=`I[U;W]`, y=`I[M;W]`, colour=Area), width=.02, height=.02)   +
  theme_bw() 
  #geom_point(data=filter(d, Language == "optimal"), aes(x=`I[M;W]`, y=`I[U;W]`), colour="red")  
ggsave("~/Downloads/efficient_deictics_1.png")

filter(d, `I[M;W]` > 1.25, `I[U;W]` < .7, Language != "simulated") %>% select(Area, Language)
filter(d, `I[M;W]` > 1.25, `I[U;W]` < .8, Language != "simulated", Area == "asia") %>%
  select(`I[M;W]` ,`I[U;W]`, Area, Language)

d$Language = substr(d$Language, 1, 5)
d$IsSim = d$Language == "simulated"
ggplot(filter(d, Language == "simulated"), aes(x=`I[M;W]`, y=`I[U;W]`)) + geom_point( colour="gray", alpha=.1) +
  geom_text_repel(data=filter(d, Language != "simulated", Language != "optimal"), aes(x=`I[M;W]`, y=`I[U;W]`, label=Language))
ggsave("~/Downloads/efficient_deictics_1_withlabels.png")

filter(d, Language == "optimal")


ggplot(filter(d, Language == "simulated"), aes(x=`grammar_complexity`, y=`I[U;W]`)) + geom_point( colour="gray", alpha=.4) +
  geom_jitter(data=filter(d, Language != "simulated", Language != "optimal"), aes(x=`grammar_complexity`, y=`I[U;W]`), width=.02, height=.02)   +
  theme_bw() #+ 
filter(d, grammar_complexity == 12) %>% arrange(`I[M;W]`)

d = separate(d, grammar_complexity, into=c("deictic", "pgs", "words"), sep="___")

d$reuse = as.numeric(d$pgs) < as.numeric(d$deictic)
ggplot(filter(d, Language == "simulated", reuse==F), aes(x=`I[M;W]`, y=`I[U;W]`, shape=reuse)) +
  geom_point( colour="gray", alpha=.4) +
  geom_point(data=filter(d, Language == "simulated", reuse==T), aes(x=`I[M;W]`, y=`I[U;W]`, shape=reuse),
             colour="pink", alpha=.4) +
  geom_jitter(data=filter(d, Language != "simulated", Language != "optimal", reuse==T),
              aes(x=`I[M;W]`, y=`I[U;W]`, shape=reuse), width=.02, height=.02, colour="red")   +
  geom_jitter(data=filter(d, Language != "simulated", Language != "optimal", reuse==F),
              aes(x=`I[M;W]`, y=`I[U;W]`, shape=reuse), width=.02, height=.02)   +
  
  theme_bw()


spanish = c("aquí
            ahí 
            allí 
            acá
            allá")
            
s = data.frame(x=1:5, count=c(161206, 42761, 26419, 8163, 12446) , 
               type=c("proximal I", "proximal II", "medial I", "medial II", "distal")) 

ggplot(s, aes(x=x, y=count)) + geom_bar(stat="identity") + theme_bw(18) + 
  theme(axis.text.x = element_text(angle=90)) + 
  scale_x_continuous(breaks=seq(1, 5), labels=s$type)
