---
title: Deictic adverbs
author: Sihan Chen, Richard Futrell, Kyle Mahowald
date: May 9 2022
---

## A walkthrough of the folders and files in the repo

---

### Folders

- ``figures``: all the figures used in the manuscript (Figures 1-8)

- ``sheets``: all data files generated by the scripts 

- ``readable_data_tables``: all the spatial deictic demonstratives complied by Nintemann et al. (2020), as spreadsheets

- ``deictic_appendix_docs``: the original appendix in Nintemann et al. (2020)

### Jupyter notebook
- ``Fit_gamma.ipynb``: the main file. It executes ``run_ib.py`` to perform the relevent calculation and generates the data files.

### Python

- ``run_ib.py``: the Python script for calculating the informativity and complexity of real lexicons and all possible lexicons, given a number of distal levels, decay parameter $\mu$, and place/goal/source coordinates

- ``ib.py``: the Python script for the Information Bottleneck (IB) method in general

- ``stirling.py``: the Python script for calculating the Stirling number of the second kind, given 2 parameters $(n, k)$. Code credit to [EXTR3ME](https://extr3metech.wordpress.com).

- ``get_prior.py``: the Python script for interpolating the prior / need probability $p(m)$ from the [Finnish corpus](https://lexiteria.com/word_frequency_list.html)

- ``get_lang_data.py``: the Python script for processing the data from Nintemann et al. (2020)

- ``enumerate_lexicons.py``: the Python script for generating all possible lexicons

- ``systematicity_combined.py``: calculate the systematicity score

### R

- ``plot_figs.R``: the R script for generating the main figures (Experiments 1 and 3)

- ``plot_grid_search.R``: the R script for analyzing the grid search data (Experiment 2)


## How to run the files

---

### Experiments 1 and 3:

1. Launch ``Fit_gamma.ipynb``. Parameters in the model are:

- ``logsp``: the trade-off parameter array. The default setting ``np.logspace(0,2, num = 1500)`` generates 1500 log-spaced tradeoff parameters $\beta$ from 1 to 100.

- ``mu``: the decay parameter. The default is 0.3.

- ``num_dists``: the number of distal levels. The default is 3.

- ``pgs_dists``: the coordinates of place/goal/source. The default is [0, 0.789, -1.315]

    The script will generate 4 files, saved in the ``sheets`` folder: 

- 2 data files for real lexicons and all possible lexicons, respectively. 
    - File 1: a data file for real lexicons, syntax: ``real_lexicons_fit_mu_<mu value>_pgs_<pgs values>num_dists_<number of distal levels>.csv``
    (the default name is ``real_lexicons_fit_mu_0.3_pgs_0_0.789_-1.315num_dists_3.csv``)
    - File 2: a data file for all possible lexicons, syntax: ``sim_lexicons_fit_mu_<mu value>_pgs_<pgs values>num_dists_<number of distal levels>.csv``
    (the default name is ``sim_lexicons_fit_mu_0.3_pgs_0_0.789_-1.315num_dists_3.csv``)

- 2 files for the deterministic and non-deterministic optimal frontier, respectively.
    - 1 data file for the deterministic optimal frontier, syntax: ``ib_curve_deter_mu_<mu value>_pgs_<pgs values>num_dists_<number of distal levels>.csv``
    - 1 data file for the non-deterministic optimal frontier, syntax: ``ib_curve_non_deter_mu_<mu value>_pgs_<pgs values>num_dists_<number of distal levels>.csv``


2. Open a terminal on the repo, run the lines below, assuming you're following the default condition
```
python systematicity_combined.py --filename sheets/real_lexicons_fit_mu_0.3_pgs_0_0.789_-1.315num_dists_3.csv
```

```
python systematicity_combined.py --filename sheets/sim_lexicons_fit_mu_0.3_pgs_0_0.789_-1.315num_dists_3.csv
```
  2 new files will be generated, with the systematicity scores attached.


3. Open and run the ``plot_figs.R`` script. Figures will be generated in the ``figures`` folder.


### Experiment 2:
1. Open a terminal on the repo, run the lines below, assuming you're following the default condition. 
    - Place/goal/source coordinate search: Run the code below (warning: this might take a day to execute. We recommend skipping this step and using our output file.). This will generate a file called ``total_grid_search_gridsearch.csv`` in the ``sheets`` folder.
```
python run_ib.py --total_search --outfile sheets/total_grid_search
```
    
    - Prior search: Run the code below. This will generate a file called ``prior_search_gridsearch.csv`` in the ``sheets`` folder.
```
python run_ib.py --prior_search --outfile sheets/prior_search
```

    - Mu search: Run the code below. This will generate a file called ``mu_search_gridsearch.csv`` in the ``sheets`` folder.
```
python run_ib.py --mu_search --outfile sheets/mu_search
```

2. Open and run the ``plot_grid_search.R`` script. Figures will be generated in the ``figures`` folder.


