library(readxl)
library(tidyverse)
library(ggrepel)
library(RColorBrewer)
library(ggnewscale)
library(latex2exp)
library(tidytext)
library(ggrepel)
library(gridExtra)
setwd("~/Documents/BCS/TedLab/here_there_way_over_there/repo_new/deictic_adverbs")
mu0 <- 0.3
prior_spec0 <- 'place_goal_source'
d <- read.csv('sheets/total_search_gridsearch.csv') %>% filter(mu == mu0, prior_spec == prior_spec0) %>%
  mutate(favoring = ifelse(abs(source) > abs(goal), 'goal', ifelse(abs(source) < abs(goal), 'source', 'neither')),
         place_location = ifelse(goal < 0 & source > 0 | goal > 0 & source < 0, 'PLACE central', 'PLACE marginal')) %>%
  mutate(favoring = factor(favoring, levels = c('goal', 'source', 'neither')),
         place_location = factor(place_location)) %>% 
  rowwise() %>% mutate(cat = paste(favoring, place_location, sep = '_')) %>% ungroup() %>%
  mutate(code = case_when(cat == 'goal_PLACE central' ~ 'S--P-G',
                           cat == 'goal_PLACE marginal' ~  'S-G-P',
                           cat == 'neither_PLACE central' ~ 'S-P-G',
                           cat == 'neither_PLACE marginal' ~ 'S=G-P',
                           cat == 'source_PLACE central' ~ 'S-P--G',
                           cat == 'source_PLACE marginal' ~ 'P-S-G'),
         code = factor(code, levels = c('S--P-G', 'S-G-P', 'S-P-G', 'S=G-P', 'S-P--G','P-S-G'))) %>%
  group_by(place_location) %>%
  mutate(lexicon_num = row_number()) %>% ungroup() %>% group_by(code) %>% mutate(mean_dist = mean(real_minus_sim_resid)) %>% ungroup() %>%
  mutate(dist_to_mean = abs(real_minus_sim_resid - mean_dist))

d_min <- d %>% slice_min(order_by = real_minus_sim_resid, n = 1) 


p3 <- ggplot(d, aes(x = code, y = real_minus_sim_resid, fill = code)) +
  stat_summary(geom = 'col', fun = 'mean', width = 0.3) +
  geom_jitter(aes(x = code, y = real_minus_sim_resid), width = 0.05, height = 0, alpha = 0.5) +
  geom_point(data = d_min, aes(x = code, y = real_minus_sim_resid), color = 'green', size = 3) +
  stat_summary(geom = 'errorbar', fun.data = 'mean_cl_boot', width = 0.3) +
  theme_classic(25) +
  xlab('Place-Goal-Source coordinates') +
  ylab('Residual Complexity') +
  # scale_fill_brewer(palette = 'Set1',
  #                   labels = unname(TeX(c('C_G < C_S < C_P', 'C_S < C_P < C_G, C_{PS} > C_{PG}', 'C_S < C_G < C_P', 'C_S < C_P < C_G, C_{PS} < C_{PG}', 'C_S < C_P < C_G, C_{PS} = C_{PG}',
  #                                        'C_S = C_G < C_P')))
  #                  ) 
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        legend.position = "none"
        ) +
  geom_hline(yintercept = 0) +
  scale_fill_manual(values = c('gray50', 'gray90', 'gray90', 'gray90','gray90','gray90'))
  # guides(fill = guide_legend(title = 'Place-Goal-Source coordinates'))
# ggsave(paste0('figures/pgs_grid_search_mu_', toString(mu0), '_', prior_spec0, '_3.png'), width = 20.3, height = 9, units = 'in')



### mu search
mus <- read.csv('sheets/mu_search_gridsearch.csv')
ggplot(mus, aes(x = mu, y = real_minus_sim_resid)) +
  geom_line() +
  geom_point() +
  theme_classic(14) +
  xlab(TeX("Decay parameter, $\\mu$")) +
  ylab(TeX('$I\\[M;W\\]_{real} - I\\[M;W\\]_{simulated}$'))
ggsave(paste0('figures/mu_grid_search_total.png'), width = 10.3, height = 5, units = 'in')

