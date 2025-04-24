import torch
import numpy as np
import random

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
        base = np.array(base)
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

def continuous_to_one_hot(seq, num_classes=4):
    one_hot_output = torch.zeros(200,num_classes)
    is_padding = False # force padding after the first predicted X
    for j, value in enumerate(seq[0]):
        if (value>=0) and not is_padding:
            if value < 0.25:
                one_hot_output[j, 0] = 1  # A
            elif value < 0.5:
                one_hot_output[j, 1] = 1  # C
            elif value < 0.75:
                one_hot_output[j, 2] = 1  # G
            else:
                one_hot_output[j, 3] = 1  # U
        else:
            one_hot_output[j, :] = -1  # Padding
            is_padding = True
    return one_hot_output


def continuous_to_one_hot_batch(generated_samples, num_classes=4):
    one_hot_output = torch.zeros(generated_samples.size(0), generated_samples.size(1), num_classes)
    for i, sample in enumerate(generated_samples):
        is_padding = False # force padding after the first predicted X
        for j, value in enumerate(sample):
            if (value>=0) and not is_padding:
                if value < 0.25:
                    one_hot_output[i, j, 0] = 1  # A
                elif value < 0.5:
                    one_hot_output[i, j, 1] = 1  # C
                elif value < 0.75:
                    one_hot_output[i, j, 2] = 1  # G
                else:
                    one_hot_output[i, j, 3] = 1  # U
            else:
                one_hot_output[i, j, :] = -1  # Padding
                is_padding = True
    return one_hot_output

def generate_latent_space_samples(batch_size, max_seq_length, device):
    random_lengths = torch.randint(30, max_seq_length + 1, (batch_size,)).to(device)
    latent_space_samples = torch.zeros((batch_size, 1, max_seq_length)).to(device)  # <--- Añadir canal

    for i, length in enumerate(random_lengths):
        latent_space_samples[i, 0, :length] = torch.rand(length.item()).to(device)   # <--- canal 0
        latent_space_samples[i, 0, length:] = -1

    return latent_space_samples, random_lengths


def generate_latent_space_samples2(batch_size, max_seq_length, device,  real_samples): # usa secuencias reales
    counts = []
    for sub_array in real_samples:
        count = 0
        for row in sub_array:
            if torch.all(row == -1):
                break
            count += 1
        counts.append(count)

    counts = np.array(counts)
    real_samples_continuous = torch.stack([
        torch.cat([x] * (200 // x.size(0)) + [x[:200 % x.size(0)]]) if x.size(0) < 200 else x[:200]
        for x in real_samples
    ]).to(device)
    real_samples_continuous = torch.stack([one_hot_to_continuous(sample) for sample in real_samples_continuous]).to(device)
    return real_samples_continuous, counts

def one_hot_to_continuous(one_hot_samples):
    continuous_output = torch.zeros(one_hot_samples.size(0))

    for i, one_hot_vector in enumerate(one_hot_samples):
        if one_hot_vector[0] == 1:  # A
            continuous_output[i] = random.uniform(0, 0.25)
        elif one_hot_vector[1] == 1:  # C
            continuous_output[i] = random.uniform(0.25, 0.5)
        elif one_hot_vector[2] == 1:  # G
            continuous_output[i] = random.uniform(0.5, 0.75)
        elif one_hot_vector[3] == 1:  # U
            continuous_output[i] = random.uniform(0.75, 1.0)
        elif torch.all(one_hot_vector == -1):  # Padding
            continuous_output[i] = -1
    
    return continuous_output

def one_hot_to_continuous_batch(seqs, max_seq_length, device):
    batch_size = len(seqs)
    continuous_batch = torch.zeros((batch_size, 1, max_seq_length), dtype=torch.float32).to(device)

    for i, seq in enumerate(seqs):
        continuous = one_hot_to_continuous(seq)  # (max_seq_length,)
        continuous_batch[i, 0, :] = continuous

    return continuous_batch

def continuous_to_rna(seq):
    one_hot = continuous_to_one_hot(seq)
    return one_hot_to_rna_with_padding(one_hot)