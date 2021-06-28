from collections import defaultdict
import pandas as pd
import re


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

def process_word(word):
    if type(word) != str:
        return [word]
    if word.startswith("1"):
        word = word[1:]
    word = re.sub("[\--=·VN<>X\*]", "", word)

    if ")" in word:
        word = [re.sub("[()]", "", word).rstrip(" "), ignore_parens(word)]
    else:
        word = [word]
    return word


def get_lang_dict(area):
    df = pd.read_excel("readable_data_tables/{}.xlsx".format(area))
    df[["Type", "Modality", "Description"]] = df.groupby(
        ["Language"]).fillna(method='ffill')[["Type", "Modality", "Description"]]
    for i in ["Place", "Goal", "Source"]:
        df[i] = [process_word(j) for j in df[i]]

    df["Type"] = [i.rstrip(" A") for i in df.Type]
    d = {lang: defaultdict(list) for lang in set(df.Language)}

    goodlangs = defaultdict(int)
    for rownum in range(df.shape[0]):
        row = df.loc[rownum]
        if row["Modality"] not in ["SI", "place"]:
            goodlangs[row["Language"]] += 1

        for j in ["Place", "Source", "Goal"]:
            d[row["Language"]][(row["Type"],
                                row["Modality"],
                                j)] += [
                i.rstrip().lstrip() for i in row[j] if type(i) == str]

    return {i: d[i] for i in d if goodlangs[i] == 0}
