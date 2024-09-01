import torch
import pandas as pd
import numpy as np
import csv
from torch import nn
from torch.optim import Adam
from utils import rna_to_one_hot, one_hot_to_rna_with_padding
from model import Generator, Discriminator
from train import train
from pretrain import pretrain_generator_as_autoencoder
import os
from datetime import datetime

if not os.path.exists('results'):
    os.makedirs('results')
log_filename = f'results/log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
real_sequences_filename = f'results/seq_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
gen_sequences_filename = f'results/gen_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parámetros
input_dim = 4  # Dimensión de la codificación one-hot
latent_dim = 100  # Dimensión del vector aleatorio de entrada

# DATOS
df = pd.read_csv('https://raw.githubusercontent.com/sinc-lab/sincFold/main/data/ArchiveII.csv')
df = df[df['len'] <= 100]
sequences = df['sequence'].tolist()
max_seq_length = max(len(seq) for seq in sequences)

# Modelos
generator = Generator(input_dim=latent_dim, output_dim=input_dim, max_seq_length=max_seq_length).to(device)
discriminator = Discriminator(input_dim=input_dim, seq_length=max_seq_length).to(device)

# Convierte todas las secuencias a One-Hot encoding y pad con -1 si es necesario
real_data = np.array([rna_to_one_hot(seq, max_seq_length) for seq in sequences])
real_data = torch.tensor(real_data, dtype=torch.float)
train_data_length = len(real_data)
train_labels = torch.zeros(train_data_length)
train_set = [(real_data[i], train_labels[i]) for i in range(train_data_length)]

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

num_epochs = 600
#loss_function = nn.BCELoss()
loss_function = nn.BCELoss(reduction='none')


mu = .1
lr = 1e-4
optimizer_discriminator = Adam(discriminator.parameters(), lr=lr)
optimizer_generator = Adam(generator.parameters(), lr=2*lr)

with open(log_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'disc_loss', 'gen_loss'])

pretrain_epochs = 100
noise_levels = [0.0, 0.1, 0.2, 0.5, 1.0]  # diferentes niveles de ruido que se van a ir introduciendo
loss_function_pretrain = nn.MSELoss()

#pretrain_generator_as_autoencoder(generator, real_data, optimizer_generator, loss_function_pretrain, pretrain_epochs, device, max_seq_length, noise_levels)
train(generator, discriminator, train_loader, loss_function, optimizer_discriminator, optimizer_generator, num_epochs, device, latent_dim, max_seq_length,mu, log_filename, real_sequences_filename, gen_sequences_filename)
