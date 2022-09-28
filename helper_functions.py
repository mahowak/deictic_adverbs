import random
import pandas as pd
import numpy as np
import re

# systematicity-related helper functions
def count_theta_patterns(key, df_wide):
    paradigm = df_wide.loc[df_wide["Language"] == key]
    # input: a pd dataframe with 3 columns
    type_list = paradigm['Type'].drop_duplicates().values
    # print(paradigm)
    # print(type_list)
    unique_patterns = 0
    max_pattern = 0
    nloops = 1 # there's no uncertainty in simulated paradigms
    # pick one row from each distal level randomly, repeat 100 times, calculate the average number of unique paradigms 
    for i in range(nloops):
        temp = pd.DataFrame({'Type':[], 'source': [], 'place': [], 'goal': []})
        # number of distal levels:
        for typ in type_list:
            # number of different paradigms in a distal level:
            nrow = paradigm.loc[paradigm['Type'] == typ].shape[0]
            # pick one randomly
            temp = pd.concat([temp, paradigm.loc[paradigm['Type'] == typ].iloc[random.randrange(nrow),]], ignore_index=True)
        # category encoding for each column
        temp['source'] = pd.factorize(temp['source'])[0]
        temp['place'] = pd.factorize(temp['place'])[0]
        temp['goal'] = pd.factorize(temp['goal'])[0]
        # convert back to np array; calculate the number of unique paradigms
        unique_patterns += len(np.unique(np.transpose(temp[["source", "place", "goal"]].to_numpy()), axis = 0))
        # if len(np.unique(np.transpose(temp[["Source", "Place", "Goal"]].to_numpy()), axis = 0)) > max_pattern:
        #     max_pattern = len(np.unique(np.transpose(temp[["Source", "Place", "Goal"]].to_numpy()), axis = 0))
    return(unique_patterns / nloops)


def count_r_patterns(key, df_wide):
    # input: a pd dataframe with 3 columns
    paradigm = df_wide.loc[df_wide["Language"] == key]
    type_list = paradigm['Type'].drop_duplicates().values
    unique_patterns = 0
    max_pattern = 0
    nloops = 1 # there's no uncertainty in simulated paradigms
    for i in range(nloops):
        temp = pd.DataFrame({'Type':[], 'source': [], 'place': [], 'goal': []})
        # print(paradigm)
        for typ in type_list:
            # number of different paradigms in a distal level:
            nrow = paradigm.loc[paradigm['Type'] == typ].shape[0]
            # pick one randomly:
            temp = pd.concat([temp, paradigm.loc[paradigm['Type'] == typ].iloc[random.randrange(nrow),]], ignore_index=True)  
            # temp = temp.append(paradigm.loc[paradigm['Type'] == typ].iloc[random.randrange(nrow),])  
        patterns = np.ones([temp.shape[0], 3])
        # print("i = ", i, ", ", temp)
        for j in range(temp.shape[0]):
            patterns[j,:] = pd.factorize(temp[["source", "place", "goal"]].iloc[j,])[0]
        # print(patterns)
        unique_patterns += len(np.unique(patterns, axis = 0))
        # if len(np.unique(patterns, axis = 0)) > max_pattern:
        #     max_pattern = len(np.unique(patterns, axis = 0))
    return(unique_patterns / nloops)


def get_entries(df):
    return(df["Language"].drop_duplicates().values)


def make_unique_paradigm_table(df_wide):
    # input: area such as africa, asia, europe, etc.
    d = df_wide
    langs = get_entries(d)
    tot = len(langs)
    score_r = []
    score_theta = []
    max_r = []
    max_theta = []
    i = 1
    for lang in langs:
        score_r = score_r + [count_r_patterns(lang, df_wide)]
        score_theta = score_theta + [count_theta_patterns(lang, df_wide)]
        if i % 500 == 1:
            print("i = ", i, " / ", tot, "; lang = ", lang)
        i += 1
    table = pd.DataFrame(list(zip(langs, score_r, score_theta)), columns=["Language", "r patterns", "theta patterns"])
    table["systematicity_score"] = table["r patterns"] + table["theta patterns"]
    # table.to_csv(outfile)
    return(table)

def paradigm_names(df):
    names = df.columns
    return(names[[bool(re.search('^D', i)) for i in df.columns]])

def systematicity(df):
    # calculate systematicity
    # distinguish all the simulated languages by giving them numbers
    k = 1
    for i in range(df.shape[0]):
        if df['Language'][i] == 'simulated':
            df['Language'][i] = df['Language'][i] + str(k)
            k += 1

    # convert the table to paradigm tables
    # id_vars = ["I[U;W]", "I[M;W]", 'MI_Objective', 'grammar_complexity', 'Language', 'Area', 'LangCategory', ]
    id_vars = ["I[U;W]", "I[M;W]", 'Language', 'Area', ]
    df_long = pd.melt(df, id_vars = id_vars,
    value_vars=paradigm_names(df), value_name='Word', var_name = 'Type')

    # split distal levels and orientations
    df_long[['Type','theta']] = df_long['Type'].str.split('_',expand=True)

    # pivot wider
    # df_wide = df_long.pivot_table(index=df_long[["I[U;W]", "I[M;W]", 'MI_Objective', 'grammar_complexity', 'Language', 'Area', 'LangCategory','Type' ]], columns = 'theta', values='Word',aggfunc='first').reset_index()
    df_wide = df_long.pivot_table(index=df_long[["I[U;W]", "I[M;W]", 'Area','Type', 'Language']], columns = 'theta', values='Word',aggfunc='first').reset_index()
    
    #make_unique_paradigm_table(df)
    df = pd.merge(df, make_unique_paradigm_table(df_wide), on = 'Language')
    return(df)

