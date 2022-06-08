library(readxl)
library(tidyverse)
library(ggrepel)
library(RColorBrewer)
library(ggnewscale)
library(latex2exp)
library(gridExtra)
library(ggrepel)


# decay parameter
mu <- 0.3
# P/G/S coordinates
pgs <- '0_0.789_-1.315'
# number of distal levels
num_dists <- 'num_dists_3'
#directory


# output file from jupyter-notebook - real lexicons
d = read.csv(paste0('sheets/real_lexicons_fit_mu_',toString(mu),'_pgs_', pgs, num_dists, '_outfile.csv')) %>% select(-Unnamed..0) %>%
  mutate(Language = gsub('\\[(.*?)\\]', '', Language)) %>% pivot_longer(2:10, names_to = 'mode', values_to = 'word') %>%
  separate(mode, into = c('distal_level', 'orientation'), sep = '_') %>% group_by(Language) %>% 
  mutate(word = as.character(word) %>% factor(levels = unique(.)) %>% as.numeric()) %>% ungroup() %>%
  unite('mode', c('distal_level', 'orientation'), sep = '_') %>%
  pivot_wider(names_from = mode, values_from = word) %>% mutate(systematicity_score = r.patterns + theta.patterns)

# output file from jupyter-notebook - simulated lexicons
d_sim = read.csv(paste0('sheets/sim_lexicons_fit_mu_',toString(mu),'_pgs_', pgs, num_dists, '_outfile.csv'))  %>% select(-Unnamed..0) %>%
  mutate(systematicity_score = r.patterns + theta.patterns)
curve_deter = read.csv(paste0('sheets/ib_curve_deter_mu_',toString(mu),'_pgs_', pgs, num_dists, '.csv'))
curve_non_deter = read.csv(paste0('sheets/ib_curve_non_deter_mu_',toString(mu),'_pgs_', pgs, num_dists, '.csv'))


# get columns indicating the (distal_level, orientation) combinations
co <- which(!is.na(str_extract(colnames(d), 'D\\d_')))
co_sim <- which(!is.na(str_extract(colnames(d_sim), 'D\\d_')))

d[, "nwords"] <- apply(d[,co], 1, max) 
d_sim[, "nwords"] <- apply(d_sim[,co_sim], 1, max) + 1
curve_deter_by_gamma <- curve_deter %>% group_by(informativity, complexity) %>% slice_min(order_by = gamma)

real_paradigm_summary <- d %>% group_by_at(vars(starts_with('D', ignore.case = FALSE))) %>% slice_sample(n=1) %>% 
  left_join(d %>% group_by_at(vars(starts_with('D', ignore.case = FALSE))) %>% summarise(n = n())) %>% ungroup() %>% mutate(Language = gsub('^\\s+', '', Language))

# overview
pos <- position_jitter(height = 0.01, width = 0.01, seed = 2)
ggplot(d, aes(x=`I.U.W.`, y=`I.M.W.`)) +
  geom_point(data = d_sim, aes(x=`I.M.W.`, y=`I.U.W.`), alpha = 0.1, size = 3, color = 'gray') +
  #geom_line(data = frontier, aes(x = complexity, y = informativity), color = 'black') +
  geom_jitter(aes(x=`I.M.W.`, y=`I.U.W.`, color = Area), size = 3, position = pos) +
  geom_line(data = curve_non_deter, aes(x = complexity, y = informativity ), color = 'black', shape = 4, size = 1) +
  geom_line(data = curve_deter, aes(x = complexity, y = informativity ), color = 'blue', shape = 4, size = 1) +
  #geom_smooth(data = d_sim, aes(x=`I.M.W.`, y=`I.U.W.`), method = 'lm')+
  theme_bw(25) +
  xlab('Complexity') +
  ylab('Informativity') +
  xlim(0,max(d$I.M.W.) + 0.15) +
  ylim(0,max(d$I.U.W.) + 0.07) +
  scale_color_brewer(palette = 'Set2') +
  geom_text_repel(data = d %>% filter(Language %in% c(#'[OC-10]  Dyirbal (Pama-Nyungan)',
                                                      'Bunoge Dogon (Dogon)',
                                                      #'[AM-21]  Kodiak Alutiiq (Eskimo-Aleut, Aleut) ',
                                                      #'Hmong Njua (Hmong-Mien, Chuanqiandian)',
                                                      #'English (Indo-European, Germanic)',
                                                      'Abau (Sepik, Upper) ',
                                                      '  Doromu-Koki (Trans-New Guinea, Manubaran) ',
                                                      'Balese (Central Sudanic)')),
                  aes(x = I.M.W., y = I.U.W., label = Language),
                  point.padding = 0.1,
                  nudge_x = -0.05,
                  nudge_y = -0.05,
                  segment.curvature = -1e-20,
                  arrow = arrow(length = unit(0.015, "npc")),
                  #position = pos
  )+
  geom_text_repel(data = d %>% filter(Language %in% c('  Dyirbal (Pama-Nyungan)',
                                                      #'Bunoge Dogon (Dogon)',
                                                      '  Kodiak Alutiiq (Eskimo-Aleut, Aleut) ',
                                                      'Hmong Njua (Hmong-Mien, Chuanqiandian)',
                                                      'English (Indo-European, Germanic)'
                                                      #'Abau (Sepik, Upper)',
                                                      )),
                  aes(x = I.M.W., y = I.U.W., label = Language),
                  point.padding = 0.1,
                  nudge_x = -0.05,
                  nudge_y = 0.05,
                  segment.curvature = -1e-20,
                  arrow = arrow(length = unit(0.015, "npc"))
  )+
  
  geom_text_repel(data = curve_deter_by_gamma %>% mutate(label = paste0('\u03b2',' = ', sprintf('%.3f',gamma))),
                  aes(x = complexity, y = informativity, label = label), color = 'blue',
                  point.padding = 0.1,
                  nudge_x = -0.02,
                  nudge_y = 0.06,
                  segment.curvature = -1e-20,
                  arrow = arrow(length = unit(0.015, "npc"))
  )+
  # geom_text_repel(data = d %>% filter(Language %in% c('[OC-10]  Dyirbal (Pama-Nyungan)',
  #                                                     'Bunoge Dogon (Dogon)',
  #                                                     '[AM-21]  Kodiak Alutiiq (Eskimo-Aleut, Aleut) ',
  #                                                     'Hmong Njua (Hmong-Mien, Chuanqiandian)',
  #                                                     'English (Indo-European, Germanic)')),
  #                 aes(x = fit_informativity_dFl_det, y = fit_complexity_dFl_det, label = Language),
  #                 point.padding = 0.1,
  #                 nudge_x = 0.05,
  #                 nudge_y = -0.1,
  #                 segment.curvature = -1e-20,
  #                 arrow = arrow(length = unit(0.015, "npc"))
  # )  +
  ggtitle('')
  #ggtitle('minimal optimality score distance, deterministic optimal q(w|m)')

ggsave(paste0('figures/Efficient_frontier_mu_',toString(mu),'_pgs_', pgs, num_dists, '.png'), width = 15.3, height = 9, units = 'in')







# color patelle for plotting
colourCount = 9
getPalette = colorRampPalette(brewer.pal(9, "Set1"))

# filter out optimal real lexicons 
real_lexicon_opts <- d %>% filter() %>% slice_min(order_by = dist_to_hull, n=5) 
real_lexicon_opts_viz <- real_lexicon_opts%>% pivot_longer(co,names_to = 'mode', values_to = 'word') %>% 
  separate(mode, into = c('distal_level', 'orientation'), sep = '_')

# filter out optimal simulated lexicons 
sim_lexicon_opts <- d_sim %>% filter() %>% slice_min(order_by = dist_to_hull,n=5) 
sim_lexicon_opts_viz <- sim_lexicon_opts %>% pivot_longer(co_sim,names_to = 'mode', values_to = 'word') %>% 
  separate(mode, into =c('distal_level', 'orientation'), sep = '_')

p2 <- ggplot(sim_lexicon_opts_viz %>% mutate(label = paste0(X, " (", 1, "), d = ", sprintf('%.4f', as.numeric(dist_to_hull)))), 
       aes(x = orientation, y = distal_level, fill = as.factor(word))) +
  facet_wrap(~reorder(label, nwords), nrow = 2, ncol = 4) +
  geom_tile() +
  scale_fill_manual(values = getPalette(colourCount)) +
  xlab('Orientation') +
  ylab('Distal Level') +
  theme_bw(17) +
  guides(fill = guide_legend(title = 'Word')) +
  ggtitle('simulated paradigms')
ggsave(paste0('figures/mu_0.3_optimal_simulated_lexicon', num_dists, '.png'), plot = p2, height = 4, width = 12, unit = 'in')


p1 <- ggplot(real_paradigm_summary %>% mutate(Language = gsub('\\((.*?)\\)', '', Language),
                                              Label = paste0(str_sub(Language, 1, -2), "(", n, "), d = ", 
                                                             sprintf('%.4f', as.numeric(dist_to_hull))))  %>% filter() %>%
               slice_min(order_by = dist_to_hull, n =5) %>% 
               pivot_longer(co, names_to = 'mode', values_to = 'word') %>%
               separate(mode, into = c('distal_level', 'orientation'), sep = '_'),  
             aes(x = orientation, y = distal_level, fill = as.factor(word))) +
  facet_wrap(~reorder(Label, nwords), nrow=2 ) +
  geom_tile() +
  scale_fill_manual(values = getPalette(colourCount)) +
  xlab('Orientation') +
  ylab('Distal Level') +
  theme_bw(17) +
  guides(fill = guide_legend(title = 'Word')) +
  ggtitle('real paradigms')
ggsave(paste0('figures/mu_', toString(mu), '_optimal_simulated_lexicon_', num_dists, '.png'), plot = p1, width = 12, height = 4, unit = 'in')

p <- grid.arrange(p1,p2, nrow=2)
ggsave(paste0('figures/mu_', toString(mu), '_optimal_lexicon_', num_dists, '.png'), plot = p, width = 15, height = 12, unit = 'in')

# opt_lex = c('Bengali (Indo-European, Indo-Iranian)',
#             'Dhimal (Sino-Tibetan, Dhimalish)',
#             'Maybrat (Maybrat-Karon) ',
#             "Tohono O'Odham  (Uto-Aztecan, Tepiman) ",
#             "Totonac, Upper Necaxa (Totonacan, Totonac) ",
#             "Maltese (Afro-Asiatic, Semitic)",
#             'Orokaiva (Trans-New Guinea, Binanderean)',
#             'Abui (Timor-Alor-Pantar, Alor) ',
#             'Guugu Yimidhirr (Pama-Nyungan, Yimidhirr-Yalanji-Yidinic)',
#             'Musqueam (Salishan, Central Salish)')

opt_lex <- (real_paradigm_summary %>% slice_min(order_by = dist_to_hull, n = 5))$Language


p4 <- ggplot(real_paradigm_summary, aes(x = dist_to_hull, y =  systematicity_score)) +
  geom_jitter(data = d_sim, aes(x = dist_to_hull, y = systematicity_score), color = 'gray', alpha = 0.5, size = 0.05) +
  geom_point(aes(size = n), position = position_jitter(width = 0, height = 0.2, seed = 3), color = 'blue') +
  facet_grid(rows = vars(nwords)) +
  #geom_point(data = real_paradigm_summary %>% filter(Language %in% opt_lex), aes(x= dist_to_hull, y = systematicity_score, color = as.factor(nwords), size = n)) +
  scale_color_manual(values = getPalette(colourCount)) +
  theme_bw(14) +
  scale_y_continuous(breaks = c(2, NA, NA, NA, 6), labels = c("systematic", "", "", "", "not systematic")) +
  scale_radius(range = c(0.05,3), breaks = c(1,10,20,30,40,50,60,70,80)) +
  xlab('Distance to the Frontier') +
  ylab('Systematicity') + 
  # geom_text(data = real_paradigm_summary %>% 
  #                   filter(Language %in% opt_lex) %>%
  #                   mutate(label = gsub('\\((.*?)\\)', '', Language),
  #                          label = tolower(str_sub(gsub(' ', '', label), 1, 3))),
  #                 aes(x = systematicity_score, y = dist_to_hull, color = as.factor(nwords), label = label), 
  #           position = position_jitter(width = 0.2, height = 0, seed = 3), hjust = -0.005, vjust = -0.5
  # ) +
  guides(color = guide_legend(title = 'Number of words'),
         size = guide_legend(title = 'Number of languages')) 
ggsave(paste0('figures/mu_', toString(mu), '_systematicity_vs_optimality_', num_dists, '.png'), width = 10, height = 5, unit = 'in', plot = p4)



# systematic and non-systematic examples
p5 <- ggplot(d %>% filter(Language == 'English (Indo-European, Germanic)') %>% 
               pivot_longer(co,names_to = 'mode', values_to = 'word') %>% 
               separate(mode, into = c('distal_level', 'orientation'), sep = '_'), 
             aes(x = orientation, y = distal_level, fill = as.factor(word))) +
  geom_tile() +
  facet_wrap(~Language) +
  scale_fill_manual(values = getPalette(colourCount)) +
  xlab('Orientation') +
  ylab('Distal Level') +
  theme_bw(17) +
  guides(fill = guide_legend(title = 'Word')) 

p6 <- ggplot(d_sim %>% filter(X == 6685) %>% 
               pivot_longer(co_sim,names_to = 'mode', values_to = 'word') %>% 
               separate(mode, into = c('distal_level', 'orientation'), sep = '_'), 
             aes(x = orientation, y = distal_level, fill = as.factor(word))) +
  geom_tile() +
  facet_wrap(~X) +
  scale_fill_manual(values = getPalette(colourCount)) +
  xlab('Orientation') +
  ylab('Distal Level') +
  theme_bw(17) +
  guides(fill = guide_legend(title = 'Word')) 

w <- grid.arrange(p5,p6, ncol=2)
ggsave(paste0('figures/mu_', toString(mu), '_suystematicity_illu_', num_dists, '.png'), plot = w, width = 12, height = 4, unit = 'in')


# pick the optimal lexicon for each tradeoff parameter
gammas <- 10^(seq(0.01, 1, 0.01))
etas <- 10^(seq(0, 1, 0.01))
opt_sys <- d_sim[1,colnames(d_sim)]
opt_sys$gammas = 0
opt_sys$etas = 0

for (i in 1:length(gammas)){
  for (j in 1:length(etas)){
    J = d_sim$I.M.W. - gammas[i] * d_sim$I.U.W. + etas[j] * d_sim$systematicity_score
    ind = which.min(J)
    opt_sys <- rbind(opt_sys, cbind(d_sim[ind,], gammas = gammas[i], etas = etas[j]))
  }
}
opt_sys = opt_sys[2:nrow(opt_sys),] %>% mutate(J = I.U.W. - gammas * I.M.W. + etas * systematicity_score)

opt_sys_real_summary <- opt_sys_real %>% group_by_at(vars(starts_with('D', ignore.case = FALSE))) %>% summarise(systematicity_score = mean(systematicity_score),
                                                                                                                gammas = min(gammas),
                                                                                                                etas = min(etas))

opt_sys_real <- d[1,colnames(d_sim)]
opt_sys_real$gammas = 0
opt_sys_real$etas = 0

for (i in 1:length(gammas)){
  for (j in 1:length(etas)){
    J = d$I.M.W. - gammas[i] * d$I.U.W. + etas[j] * d$systematicity_score
    ind = which.min(J)
    opt_sys_real <- rbind(opt_sys_real, cbind(d[ind,], gammas = gammas[i], etas = etas[j]))
  }
}
opt_sys_real = opt_sys_real[2:nrow(opt_sys_real),] %>% mutate(J = I.U.W. - gammas * I.M.W. + etas * systematicity_score)

opt_sys_summary <- opt_sys %>% group_by_at(vars(starts_with('D', ignore.case = FALSE))) %>% summarise(systematicity_score = mean(systematicity_score),
                                                                                                      gammas = min(gammas),
                                                                                                      etas = min(etas))

ggplot(opt_sys_summary %>% 
         pivot_longer(1:9, names_to = 'mode', values_to = 'word') %>%
         separate(mode, into = c('distal_level', 'orientation'), sep = '_') %>% 
         mutate(label = paste0('beta = ', sprintf('%.3f', as.numeric(gammas)), '; gamma = ', sprintf('%.3f', as.numeric(etas)))), 
       aes(x = orientation, y = distal_level, fill = as.factor(word))) +
  facet_wrap(~label) +
  geom_tile() +
  scale_fill_manual(values = getPalette(colourCount)) +
  xlab('Orientation') +
  ylab('Distal Level') +
  theme_bw(17) +
  guides(fill = guide_legend(title = 'Word'))

ggsave(paste0('figures/mu_', toString(mu), '_systematicity_and_optimal_lexicons_pgs_', pgs, num_dists, '.png'), width = 10, height = 6, unit = 'in')
