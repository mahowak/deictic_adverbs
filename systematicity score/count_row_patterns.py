import pandas as pd
import numpy as np

""" 
This script tries to calculate the total number of distinct row paradigms.

Input: the same form as the example paradigm on Overleaf

example input 1:
Theta \ R   0   1   2
    -1      A   B   B
     0      C   D   D
     1      C   D   D

The total number of distinct row paradigms is 1 (aka for each row, it is word1, word2, word2)

example input 2:
Theta \ R   0   1   2
    -1      A   B   B
     0      C   C   D
     1      C   D   D

The total number of distinct row paradigms is 2 (aka for each row, there is word1, word2, word2, as well as word1, word1, word2)

"""

def count_row_patterns(paradigm):
    # input: an numpy array with 3 rows
    # convert np array to pd array, with each column representing each theta attribute
    paradigm_df = pd.DataFrame(np.transpose(paradigm), columns=['row1', 'row2', 'row3'])
    # category encoding for each column
    paradigm_df['row1'] = paradigm_df['row1'].astype('category').cat.codes
    paradigm_df['row2'] = paradigm_df['row2'].astype('category').cat.codes
    paradigm_df['row3'] = paradigm_df['row3'].astype('category').cat.codes
    # convert back to np array; calculate the number of unique paradigms
    unique_patterns = len(np.unique(np.transpose(paradigm_df.values), axis = 0))
    return(unique_patterns)

a = np.array([['A', 'B', 'B'], ['C', 'D', 'D'], ['C', 'D', 'D']])
print("the number of unique row paradigms is: ", count_row_patterns(a))
