import os
import torch
import pandas as pd
import numpy as np
import csv
from torch import nn
from torch.optim import Adam, SGD
from datetime import datetime

from utils import rna_to_one_hot, one_hot_to_rna_with_padding
from model import Discriminator,GeneratorCNNOrdinal
from utils import generate_latent_space_samples, one_hot_to_continuous_batch,one_hot_to_rna_with_padding, continuous_to_rna
from model import padding_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# LOGs
if not os.path.exists('results'): os.makedirs('results')
log_filename = f'results/log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
real_seq_filename = f'results/seq_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
losses_filename = f'results/losses_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
generated_seq_filename = f'results/generated_seq_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
with open(log_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'disc_loss', 'gen_loss','Train Disc/Gen'])
real_seq_logfile = open(real_seq_filename, mode='a', newline='')
real_seq_logger = csv.writer(real_seq_logfile) 
real_seq_logger.writerow(["epoch", "batch", "seq_in_batch", "sequence", "loss"]) 
generated_seq_logfile = open(generated_seq_filename, mode='a', newline='')
generated_seq_logger = csv.writer(generated_seq_logfile) 
generated_seq_logger.writerow(["epoch", "discriminator_batch", "sequence", "loss"]) 

# DATA
df = pd.read_csv('./ArchiveII.csv')
df = df[df['len'] <= 200]
sequences = df['sequence'].head(40).tolist()
max_seq_length = 200

# convertir las secuencias a one-hot y padding
real_data = np.array([rna_to_one_hot(seq, max_seq_length) for seq in sequences])
real_data = torch.tensor(real_data, dtype=torch.float)
train_data_length = len(real_data)
train_labels = torch.zeros(train_data_length)
train_set = [(real_data[i], train_labels[i]) for i in range(train_data_length)]

# PARAMS
latent_dim = 200  # Dimensión del vector aleatorio de entrada
batch_size = 20 # 32
num_epochs = 500
bce_loss = nn.BCELoss() # reduction='mean' by default

lr_dis = 1e-5 #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
lr_gen = 1e-3

# MODELS
generator = GeneratorCNNOrdinal(input_dim=latent_dim, 
                                output_dim=1, 
                                max_seq_length=max_seq_length,
                                initial_weights=[0.0, 1.0, 0.0]).to(device)
discriminator = Discriminator(input_dim=1, seq_length=max_seq_length).to(device)

optimizer_discriminator = Adam(discriminator.parameters(), lr=lr_dis)
optimizer_generator = Adam(generator.parameters(), lr=lr_gen)

# TRAINING
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

last_discriminator_loss = None
last_generator_loss = None
discriminator_losses = []
generator_losses = []

for epoch in range(num_epochs):
    discriminator.train()
    generator.train()

    epoch_discriminator_losses = []
    epoch_generator_losses = []

    for n, (real_samples, _) in enumerate(train_loader):

        # Discriminator =====================================
        optimizer_discriminator.zero_grad()

        real_samples = one_hot_to_continuous_batch(real_samples, max_seq_length, device)
        real_samples_labels = torch.ones((real_samples.size(0), 1)).to(device) 

        with torch.no_grad(): generated_samples = generator(real_samples)
        generated_samples_labels = torch.zeros((real_samples.size(0), 1)).to(device)
     
        output_discriminator_fake = discriminator(generated_samples)
        output_discriminator_real = discriminator(real_samples)
        loss_discriminator = bce_loss(output_discriminator_real, real_samples_labels) + \
                             bce_loss(output_discriminator_fake, generated_samples_labels)

        loss_discriminator.backward() #retain_graph=True)
        optimizer_discriminator.step()



        # Generator =====================================
        optimizer_generator.zero_grad()

        generated_samples = generator(real_samples)
        output_discriminator_fakereal = discriminator(generated_samples)
        fakereal_samples_labels = torch.ones((real_samples.size(0), 1)).to(device) 

        loss_generator = bce_loss(output_discriminator_fakereal, fakereal_samples_labels)

        loss_generator.backward()
        optimizer_generator.step()
    


        # Prints and logs ========================================
        epoch_discriminator_losses.append(loss_discriminator.item())
        epoch_generator_losses.append(loss_generator.item())

        if epoch % 10 == 0:
            # log losses y sequences
            for i in range(len(generated_samples)): # sequences in the batch
                # log the input to the generator
                real_rna_seq = continuous_to_rna(real_samples[i].cpu().detach().numpy())
                generated_seq_logger.writerow([epoch, n, i, real_rna_seq, "-"])
                
                # log the generated sequence
                generated_rna_seq = continuous_to_rna(generated_samples[i].cpu().detach().numpy())  
                generated_seq_logger.writerow([epoch, n, "_", generated_rna_seq, f"{loss_generator.item():.2f}"])                          
                
                # fast check console printing
                if n == 0 and i == 0:
                    print(f"input {real_rna_seq[:33]} ... {real_rna_seq[-33:]}")
                    print(f"gener {generated_rna_seq[:33]} ... {generated_rna_seq[-33:]}")

            # log secuencias reales
            for i in range(len(generated_samples)):
                real_rna_seq = continuous_to_rna(real_samples[i].cpu().detach().numpy())
                real_seq_logger.writerow([epoch, n, i, real_rna_seq, "-"])

    discriminator_losses.extend(epoch_discriminator_losses)
    generator_losses.extend(epoch_generator_losses)

    # mostrar pérdidas
    last_d = epoch_discriminator_losses[-1] if epoch_discriminator_losses else 0
    last_g = epoch_generator_losses[-1] if epoch_generator_losses else 0
    print(f"Epoch {epoch} - Loss discriminator: {last_d:<.6f} | Loss generator: {last_g:<.6f}")# | Training: {train_phase}")

    with open(log_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, f"{loss_discriminator.item():.6f}", f"{loss_generator.item():.6f}"])#, train_phase])

real_seq_logfile.close()

filtro_filename = "filtro.txt"
filtros = filtros = generator.conv.weight.detach().cpu().numpy()
with open(filtro_filename, "w") as file:
    for i, filtro in enumerate(filtros):
        file.write(f"Filtro {i}:\n")
        file.write(str(filtro) + "\n\n")