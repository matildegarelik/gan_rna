import torch
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
        max_value = np.max(base)
        if max_value > 0.5:
            idx = np.argmax(base)
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

def continuous_to_one_hot(generated_samples, num_classes=4):
    one_hot_output = torch.zeros(generated_samples.size(0), generated_samples.size(1), num_classes)
    for i, sample in enumerate(generated_samples):
        for j, value in enumerate(sample):
            if value>=0:
                if value < 0.25:
                    one_hot_output[i, j, 0] = 1  # A
                elif value < 0.5:
                    one_hot_output[i, j, 1] = 1  # C
                elif value < 0.75:
                    one_hot_output[i, j, 2] = 1  # G
                elif value <= 1.0:
                    one_hot_output[i, j, 3] = 1  # U
            else:
                one_hot_output[i, j, :] = -1  # Padding
    return one_hot_output

def generate_latent_space_samples(batch_size, max_seq_length, device):
    random_lengths = torch.randint(30, max_seq_length + 1, (batch_size,)).to(device)
    latent_space_samples = torch.zeros((batch_size, max_seq_length)).to(device)
    for i, length in enumerate(random_lengths):
        latent_space_samples[i, :length] = torch.rand(length.item()).to(device)
        latent_space_samples[i, length:] = -1
    return latent_space_samples, random_lengths