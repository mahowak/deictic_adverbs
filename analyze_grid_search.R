library(tidyverse)

d = read_delim("run_grid_search_pgs.csv", delim=" ",
               col_names = c("place", "goal", "source", "m1", "m2", "diff"))

arrange(d, -diff)
arrange(d, diff)


d = read_delim("run_grid_search_optimal.csv", delim=" ",
               col_names = c("place", "goal", "source", "real", "simulated",
                             "optimal", "diff"))
d$realvsoptimal = d$real - d$optimal

arrange(d, -realvsoptimal)
arrange(d, realvsoptimal)


#########
d = read_delim("run_grid_search_optimal_0801_resid.csv", delim=" ",
               col_names = c("place", "goal", "source", "real", "simulated",
                             "optimal", "real_sd", "optimal_sd",
                             "simulated_sd", "diff", "resid"))
d$realvssim.z = (d$real - d$simulated)/d$simulated_sd
d$real.vs.optimal = (d$real - d$optimal) #/d$simulated_sd
d$score = d$real.vs.optimal/d$realvssim.z

arrange(d, -realvssim.z) %>%
  select(place:source, realvssim.z)

arrange(d, realvssim.z) %>%
  select(place:source, realvssim.z)

arrange(d, real.vs.optimal) %>%
  select(place:source, real.vs.optimal)

arrange(d, -real.vs.optimal) %>%
  select(place:source, real.vs.optimal)

arrange(d, resid) %>%
  select(place:source, resid)


arrange(d, -realvsoptimal)
arrange(d, realvsoptimal)

arrange(d, score)
arrange(d, -score)

####
d = read_csv("outfiles/run_grid_search_optimal_0803_gridsearch.csv")
arrange(d, real_minus_opt_resid_sq) %>%
  select(place:source, real_minus_opt_resid, real_minus_opt_resid_sq)
arrange(d, -real_minus_opt_resid_sq) %>%
  select(place:source, real_minus_opt_resid, real_minus_opt_resid_sq)


#######
d = read_csv("outfiles/run_grid_search_optimal_0803_zscore_gridsearch.csv")
arrange(d, real_minus_opt_resid_sq) %>%
  select(place:source, real_minus_opt_resid, real_minus_opt_resid_sq)
arrange(d, -real_minus_opt_resid_sq) %>%
  select(place:source, real_minus_opt_resid, real_minus_opt_resid_sq)

########
d = read_csv("outfiles/run_grid_search_optimal_trysim_gridsearch.csv")
arrange(d, real_minus_sim_resid_sq) %>%
  select(place:source, real_minus_sim_resid_sq)
arrange(d, -real_minus_sim_resid_sq) %>%
  select(place:source, real_minus_sim_resid_sq)


###########
d = read_csv("outfiles/test_hull_gridsearch.csv")
arrange(d, dist_to_hull) %>%
  select(place:source, dist_to_hull)
arrange(d, -dist_to_hull) %>%
  select(place:source, dist_to_hull)

ggplot(filter(d, dist_to_hull < 18,
              place < 2, source > -2), aes(x=goal, y=source, colour=dist_to_hull)) +
  geom_point() +
  scale_colour_gradient(low="black", high="white", limits=c(4, 18)) + 
  xlim(0, 2) + 
  ylim(-2, 0) + 
  scale_y_reverse() + 
  theme_bw()
ggsave("hull_search.png")

########
###########
d = read_csv("outfiles/test_hull_wide_gridsearch.csv")
arrange(d, dist_to_hull) %>%
  select(place:source, dist_to_hull)
arrange(d, -dist_to_hull) %>%
  select(place:source, dist_to_hull)

ggplot(filter(d, dist_to_hull < 15,
              place < 2, source > -2), aes(x=goal, y=source, colour=dist_to_hull)) +
  geom_point() +
  scale_colour_gradient(low="black", high="white", limits=c(4, 15)) + 
  scale_y_reverse() + 
  theme_bw()


