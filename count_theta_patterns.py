import pandas as pd
import numpy as np
from collections import defaultdict
import re

""" 
This script tries to calculate the total number of distinct theta paradigms (number of distinct paradigms if the theta value varies).

Input: the transposed form of the example paradigm on Overleaf

example input 1:
R \ theta  -1   0   1
     0      A   B   B
     1      C   D   D
     2      C   D   D

The total number of distinct theta paradigms is 1 (aka for each theta, it is word1, word2, word2)

example input 2:
R \ theta  -1   0   1
     0      A   B   B
     1      C   B   D
     2      C   D   D

The total number of distinct theta paradigms is 2 (aka for each theta, there is word1, word2, word2, as well as word1, word1, word2)

"""

def count_theta_patterns(area, key):
    # input: a pd dataframe with 3 columns
    paradigm = get_paradigm_table(area, key)
    # print(paradigm)
    # category encoding for each column
    paradigm['Source'] = pd.factorize(paradigm['Source'])[0]
    paradigm['Place'] = pd.factorize(paradigm['Place'])[0]
    paradigm['Goal'] = pd.factorize(paradigm['Goal'])[0]
    # print(paradigm)
    # # convert back to np array; calculate the number of unique paradigms
    unique_patterns = len(np.unique(np.transpose(paradigm.to_numpy()), axis = 0))
    # print(np.sort(np.transpose(paradigm.to_numpy())))
    return(unique_patterns)

def count_r_patterns(area, key):
    # input: a pd dataframe with 3 columns
    paradigm = get_paradigm_table(area, key)
    # print(paradigm)
    # category encoding for each column
    patterns = np.ones([paradigm.shape[0], 3])
    for i in range(paradigm.shape[0]):
        patterns[i,:] = pd.factorize(paradigm.iloc[i,])[0]
    # print(patterns)
    unique_patterns = len(np.unique(patterns, axis = 0))
    # print(np.sort(patterns))
    return(unique_patterns)

    


# from Kyle
def ignore_parens(s):
    ignore = False
    newword = ""
    for i in s:
        if i == "(":
            ignore = True
        if ignore == False:
            newword += i
        if i == ")":
            ignore = False
    return newword.lstrip(" ").rstrip(" ")


# from Kyle
def process_word(word):
    if type(word) != str:
        return word
    if word.startswith("1"):
        word = word[1:]
    word = re.sub("[\--=·VN<>X\*]", "", word)

    if ")" in word:
        word = re.sub("[()]", "", word).rstrip(" "), ignore_parens(word)
    else:
        word = word
    return word


# modified from Kyle
def get_lang_dict(area):
    df = pd.read_excel("readable_data_tables/{}.xlsx".format(area))
    df[["Type", "Modality", "Description"]] = df.groupby(
        ["Language"]).fillna(method='ffill')[["Type", "Modality", "Description"]]
    for i in ["Place", "Goal", "Source"]:
        df[i] = [process_word(j) for j in df[i]]

    df["Type"] = [i.rstrip(" A") for i in df.Type]

    return df

def get_entries(df):
    return(df["Language"].drop_duplicates().values)

def get_paradigm_table(area, key):
    # area: the excel sheet name ('africa', 'asia', etc.)
    # key: the 'Language' column entry in each excel sheet
    df = get_lang_dict(area)
    df_lang = df.loc[df["Language"] == key]
    df_paradigm = df_lang.loc[df_lang['Type'] != 'SI']
    return(df_paradigm[["Source", "Place", "Goal"]])

def test_empty_entries(paradigm):
    return(paradigm.isnull().values.any())

def make_unique_theta_paradigm_table(area):
    # input: area such as africa, asia, europe, etc.
    d = get_lang_dict(area)
    langs = get_entries(d)
    score = []
    is_weird = []
    for lang in langs:
        score = score + [count_theta_patterns(area, lang)]
        is_weird = is_weird + [test_empty_entries(get_paradigm_table(area, lang))]
    df = pd.DataFrame(list(zip(langs, score, is_weird)), columns=["Language", "theta patterns", "duplicated?"])
    return(df)

def make_unique_r_paradigm_table(area):
    # input: area such as africa, asia, europe, etc.
    d = get_lang_dict(area)
    langs = get_entries(d)
    score = []
    is_weird = []
    for lang in langs:
        score = score + [count_r_patterns(area, lang)]
        is_weird = is_weird + [test_empty_entries(get_paradigm_table(area, lang))]
    df = pd.DataFrame(list(zip(langs, score, is_weird)), columns=["Language", "r patterns", "duplicated?"])
    return(df)

def make_unique_paradigm_table(area):
    # input: area such as africa, asia, europe, etc.
    d = get_lang_dict(area)
    langs = get_entries(d)
    score_r = []
    score_theta = []
    is_weird = []
    for lang in langs:
        score_r = score_r + [count_r_patterns(area, lang)]
        score_theta = score_theta + [count_theta_patterns(area, lang)]
        is_weird = is_weird + [test_empty_entries(get_paradigm_table(area, lang))]
    df = pd.DataFrame(list(zip(langs, score_r, score_theta, is_weird)), columns=["Language", "r patterns", "theta_patterns", "duplicated?"])
    return(df)


# example
# print(count_r_patterns('europe', 'Russian (Indo-European, Slavic)'))
print(make_unique_paradigm_table('europe'))
