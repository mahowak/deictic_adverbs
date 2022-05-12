import random
import itertools
import numpy as np

flat = itertools.chain.from_iterable

def unique(list1):
    # initialize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    # print list
    return unique_list


def integer_partitions(n, k, l=1):
    '''n is the integer to partition, k is the length of partitions, l is the min partition element size'''
    # From https://stackoverflow.com/questions/18503096/python-integer-partitioning-with-given-k-partitions user Snakes and Coffee
    if k < 1:
        return
    elif k == 1:
        if n >= l:
            yield (n,)
        return
    else:
        for i in range(l, n+1):
            for result in integer_partitions(n-i, k-1, i):
                yield (i,) + result

def expansions(xs, k):
    for partition in integer_partitions(k, len(xs)):
        for partition_perm in set(itertools.permutations(partition)):
            expanded = [[x]*n for x, n in zip(xs, partition_perm)]
            yield tuple(flat(expanded))

def onto_mappings(domain, codomain):
    return set(flat(map(itertools.permutations, expansions(range(codomain), domain))))

def remove_duplicates(mapping):
    m_new = []
    for m in mapping:
        if unique(m) == sorted(unique(m)):
            m_new.append(m)
    return m_new

def to_stochastic_matrix(mapping):
    """ Input:
    mapping: a tuple such as (1,3,0,1) which means the first meaning is mapped to word 1, the second to word 3, the third to word 3, etc.

    Output:
    A stochastic matrix giving p(word|meaning) according to the mapping
    """
    mapping = tuple(mapping)
    X = len(mapping)
    Y = len(set(mapping))
    a = np.zeros((X, Y))
    for x, y in enumerate(mapping):
        a[x,y] = 1
    return a

def enumerate_possible_lexicons(num_meanings, num_words):
    """ Enumerate all possible lexicons (mappings from meanings to words) in terms of p(w|m) """
    for mapping in remove_duplicates(onto_mappings(num_meanings, num_words)):
        yield mapping, to_stochastic_matrix(mapping)

def get_random_lexicon(num_meanings, num_words, seed=None):
    words = list(range(num_words))
    random.seed(seed)
    words += random.choices(words, k=num_meanings - num_words)
    random.seed(seed)
    words = random.sample(words, len(words))
    a = np.zeros([len(words), len(set(words))])
    for x, y in enumerate(words):
        a[x, y] = 1
    return (words, a)


