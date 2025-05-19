import torch
import pandas as pd
import numpy as np
import csv
from torch import nn
from torch.optim import Adam, SGD
from utils import rna_to_one_hot, one_hot_to_rna_with_padding
from model import Discriminator,GeneratorCNNOrdinal
from train import train
import os
from datetime import datetime

# archivos log
if not os.path.exists('results'):
    os.makedirs('results')
log_filename = f'results/log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
real_sequences_filename = f'results/seq_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
losses_filename = f'results/losses_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
generated_seq_filename = f'results/generated_seq_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

with open(log_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'disc_loss', 'gen_loss','Train Disc/Gen'])

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# DATOS
df = pd.read_csv('./ArchiveII.csv')
df = df[df['len'] <= 200]
sequences = df['sequence'].head(10).tolist()
max_seq_length = 200

# convertir las secuencias a one-hot y padding
real_data = np.array([rna_to_one_hot(seq, max_seq_length) for seq in sequences])
real_data = torch.tensor(real_data, dtype=torch.float)
train_data_length = len(real_data)
train_labels = torch.zeros(train_data_length)
train_set = [(real_data[i], train_labels[i]) for i in range(train_data_length)]

# parámetros
latent_dim = 200  # Dimensión del vector aleatorio de entrada
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
num_epochs = 100
loss_function = nn.BCELoss()
mu = .1
lr = 0.1

# modelos
generator = GeneratorCNNOrdinal(input_dim=latent_dim, output_dim=1, max_seq_length=max_seq_length).to(device)
discriminator = Discriminator(input_dim=1, seq_length=max_seq_length).to(device)
optimizer_discriminator = Adam(discriminator.parameters(), lr=lr/10)
optimizer_generator = Adam(generator.parameters(), lr=lr)

train(generator, discriminator, train_loader, loss_function, optimizer_discriminator, optimizer_generator, num_epochs, device, latent_dim, max_seq_length,mu, log_filename, real_sequences_filename,generated_seq_filename, losses_filename)

import torch

filtro_filename = "filtro.txt"
filtros = filtros = generator.conv.weight.detach().cpu().numpy()
with open(filtro_filename, "w") as file:
    for i, filtro in enumerate(filtros):
        file.write(f"Filtro {i}:\n")
        file.write(str(filtro) + "\n\n")