import numpy as np

def rna_to_one_hot(seq, max_length):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'U': 3, "X": 4}
    one_hot = np.zeros((max_length, 5))  # inicializar con ceros
    
    for idx, base in enumerate(seq):
        one_hot[idx, mapping[base]] = 1
        
    # relleno de padding para la longitud restante
    if len(seq) < max_length:
        one_hot[len(seq):] = [0, 0, 0, 0, 1]

    return one_hot

def one_hot_to_rna_with_padding(one_hot_seq):
    mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'U', 4: "X"}
    seq = ""
    for base in one_hot_seq:
        idx = np.argmax(base)
        seq += mapping[idx]
        
    return seq


