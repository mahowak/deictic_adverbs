import itertools

import numpy as np

flat = itertools.chain.from_iterable

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

def to_stochastic_matrix(mapping):
    mapping = tuple(mapping)
    X = len(mapping)
    Y = len(set(mapping))
    a = np.zeros((X, Y))
    for x, y in enumerate(mapping):
        a[x,y] = 1
    return a

def enumerate_possible_lexicons(num_meanings, num_words):
    """ Enumerate all possible lexicons (mappings from meanings to words) in terms of p(w|m) """
    mappings = onto_mappings(num_meanings, num_words)
    return map(to_stochastic_matrix, mappings)