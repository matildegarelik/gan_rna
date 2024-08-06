import numpy as np

def rna_to_one_hot(seq, max_length):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    one_hot = np.zeros((max_length, 4))  # inicializar con ceros
    
    for idx, base in enumerate(seq):
        if base in mapping:
            one_hot[idx, mapping[base]] = 1
        elif base == 'X':
            one_hot[idx] = [-1, -1, -1, -1]  
    
    # relleno de padding para la longitud restante
    if len(seq) < max_length:
        one_hot[len(seq):] = [-1, -1, -1, -1]

    return one_hot

def one_hot_to_rna_with_padding(one_hot_seq):
    mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}
    seq = ""
    for base in one_hot_seq:
        if np.array_equal(base, [-1, -1, -1, -1]):
            seq += 'X'
        else:
            idx = np.argmax(base)
            if base[idx] == 1:
                seq += mapping[idx]
            else:
                seq += 'X'
    return seq

def one_hot_to_rna(one_hot_seq):
    mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}
    seq = ""
    for base in one_hot_seq:
        idx = np.argmax(base)
        seq += mapping[idx]
    return seq
