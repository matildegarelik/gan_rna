import torch
import pandas as pd
import numpy as np
import csv
from torch import nn
from torch.optim import Adam, SGD
from utils import rna_to_one_hot, one_hot_to_rna_with_padding
from model import Generator, Discriminator, GeneratorCNN
from train import train
from pretrain import pretrain_generator_as_autoencoder
import os
from datetime import datetime

if not os.path.exists('results'):
    os.makedirs('results')
log_filename = f'results/log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
real_sequences_filename = f'results/seq_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
losses_filename = f'results/losses_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
generated_seq_filename = f'results/generated_seq_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parámetros
input_dim = 4  # Dimensión de la codificación one-hot
latent_dim = 200  # Dimensión del vector aleatorio de entrada

# DATOS
df = pd.read_csv('https://raw.githubusercontent.com/sinc-lab/sincFold/main/data/ArchiveII.csv')
df = df[df['len'] <= 200]
sequences = df['sequence'].tolist()
#max_seq_length = max(len(seq) for seq in sequences)
max_seq_length = 200

# Modelos
generator = GeneratorCNN(input_dim=latent_dim, output_dim=input_dim, max_seq_length=max_seq_length).to(device)
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
lr = 5e-5
optimizer_discriminator = Adam(discriminator.parameters(), lr=lr)
optimizer_generator = Adam(generator.parameters(), lr=lr)

#optimizer_discriminator = SGD(discriminator.parameters(), lr=lr)
#optimizer_generator = SGD(generator.parameters(), lr=lr)

with open(log_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'disc_loss', 'gen_loss','Train Disc/Gen'])

pretrain_epochs = 100
noise_levels = [0.0, 0.1, 0.2, 0.5, 1.0]  # diferentes niveles de ruido que se van a ir introduciendo
loss_function_pretrain = nn.MSELoss()

#pretrain_generator_as_autoencoder(generator, real_data, optimizer_generator, loss_function_pretrain, pretrain_epochs, device, max_seq_length, noise_levels)
train(generator, discriminator, train_loader, loss_function, optimizer_discriminator, optimizer_generator, num_epochs, device, latent_dim, max_seq_length,mu, log_filename, real_sequences_filename,generated_seq_filename, losses_filename)

import torch

filtro_filename = "filtro.txt"
filtros = generator.conv[0].weight.detach().cpu().numpy()
with open(filtro_filename, "w") as file:
    for i, filtro in enumerate(filtros):
        file.write(f"Filtro {i}:\n")
        file.write(str(filtro) + "\n\n")

"""import matplotlib.pyplot as plt
import numpy as np

# Guardar la matriz de pesos en un archivo de texto
pesos_filename = "pesos.txt"
with open(pesos_filename, "w") as f:
    for i, layer in enumerate(generator.model):
        if isinstance(layer, torch.nn.Linear):
            f.write(f"Layer {i} - Linear\n")
            f.write("Weights:\n")
            # Convertir los pesos a numpy y guardarlos
            weights = layer.weight.detach().cpu().numpy()
            f.write(str(weights) + "\n")
            f.write("Bias:\n")
            bias = layer.bias.detach().cpu().numpy()
            f.write(str(bias) + "\n")
            f.write("="*50 + "\n")

            # Graficar la matriz de pesos
            plt.figure(figsize=(10, 8))
            plt.imshow(weights, cmap='viridis', aspect='auto')  # Usar un mapa de colores adecuado
            plt.colorbar()
            plt.title(f'Matriz de Pesos - Layer {i}')
            plt.xlabel('Neuronas de la capa')
            plt.ylabel('Pesos de la capa')
            plt.tight_layout()

            # Guardar la gráfica como imagen PNG
            plt.savefig(f'pesos_layer_{i} (solo loss mse, CNN).png')
            plt.close()  # Cerrar la figura para liberar memoria"
"""