import torch
from torch import nn
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(111)

# HACE PADDING CON -1 HASTA MAX_SEQ_LENGHT
def rna_to_one_hot(seq, max_length):
    # Mapeo de nucleótidos y padding a índices
    mapping = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'X': -1}
    one_hot = np.full((max_length, 4), -1)  # Inicializar con -1 para el padding
    
    for idx, base in enumerate(seq):
        if base in mapping and mapping[base] != -1:
            one_hot[idx, mapping[base]] = 1
        elif base == 'X':
            one_hot[idx] = [-1, -1, -1, -1]  # Relleno con -1 para 'X' (padding)
    
    return one_hot

def one_hot_to_rna_with_padding(one_hot_seq):
    mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}
    seq = ""
    for base in one_hot_seq:
        if np.array_equal(base, [-1, -1, -1, -1]):
            seq += 'X'  # Padding
        else:
            idx = np.argmax(base)
            if base[idx] == 1:
                seq += mapping[idx]
            else:
                seq += 'X'  # Tratamiento de padding en caso de error
    return seq

def one_hot_to_rna(one_hot_seq):
    mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}
    seq = ""
    for base in one_hot_seq:
        idx = np.argmax(base)
        seq += mapping[idx]
    return seq

class Discriminator(nn.Module):
    def __init__(self, input_dim, seq_length):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(seq_length * input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.seq_length = seq_length
        self.input_dim = input_dim

    def forward(self, x):
        # Se convierten a -1 las posiciones de padding para que no contribuyan a la salida del discriminador
        x = x.view(-1, self.seq_length * self.input_dim)
        padding_mask = (x != -1).float()
        x = x * padding_mask  # Aplica máscara para ignorar los padding (-1)
        return self.model(x)
        
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, max_seq_length):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim+1, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, max_seq_length * output_dim)
        )
        self.max_seq_length = max_seq_length
        self.output_dim = output_dim

    def forward(self, x, lengths):
        # Agregar longitud a la entrada
        lengths = lengths.unsqueeze(1).float() / self.max_seq_length 
        x = torch.cat([x, lengths], dim=1)
        output = self.model(x)
        output = output.view(-1, self.max_seq_length, self.output_dim)

        max_indices = torch.argmax(output, dim=-1) #  encuentra los indices con los valores maximos para cada posición en la secuencia.
        one_hot_output = F.one_hot(max_indices, num_classes=self.output_dim).float()
        return one_hot_output
        

# Parámetros
input_dim = 4 # Dimensión de la codificación one-hot
latent_dim = 100 # Dimensión del vector aleatorio de entrada


# DATOS 
df = pd.read_csv('https://raw.githubusercontent.com/sinc-lab/sincFold/main/data/ArchiveII.csv')
df=df[df['len']<=100]
sequences = df['sequence'].tolist()
max_seq_length = max(len(seq) for seq in sequences)

generator = Generator(input_dim=latent_dim, output_dim=input_dim, max_seq_length=max_seq_length).to(device)
discriminator = Discriminator(input_dim=input_dim, seq_length=max_seq_length).to(device)

# Convierte todas las secuencias a One-Hot encoding y pad con ceros si es necesario
real_data = np.array([rna_to_one_hot(seq, max_seq_length) for seq in sequences])
real_data = torch.tensor(real_data, dtype=torch.float)
train_data_length= len(real_data)
train_labels = torch.zeros(train_data_length)
train_set = [
    (real_data[i], train_labels[i]) for i in range(train_data_length)
]

batch_size = 10
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)

num_epochs = 600
loss_function = nn.BCELoss()

lr = 0.0002
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr*10)

with open('GAN_log.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch','disc_loss','gen_loss'])

# ENTRENAMIENTO
for epoch in range(num_epochs):
    for n, (real_samples, _) in enumerate(train_loader):
        real_samples = real_samples.to(device)
        real_samples_labels = torch.ones((real_samples.size(0), 1)).to(device) * 0.9  # Label smoothing

        # Generar muestras falsas
        latent_space_samples = torch.randn((real_samples.size(0), latent_dim)).to(device)
        random_lengths = torch.randint(1, max_seq_length + 1, (real_samples.size(0),)).to(device)
        generated_samples = generator(latent_space_samples, random_lengths)

        generated_samples_labels = torch.zeros((real_samples.size(0), 1)).to(device)
        
        # Mezclar datos reales y generados
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

        # Entrenamiento del discriminador
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(output_discriminator, all_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()

    # Entrenamiento del generador
    for n, (real_samples, _) in enumerate(train_loader):
        real_samples = real_samples.to(device)
        real_samples_labels = torch.ones((real_samples.size(0), 1)).to(device)

        # Datos para entrenar el generador
        latent_space_samples = torch.randn((real_samples.size(0), latent_dim)).to(device)
        random_lengths = torch.randint(1, max_seq_length + 1, (real_samples.size(0),)).to(device)
        generated_samples = generator(latent_space_samples, random_lengths)
        
        # Salida del discriminador para las muestras generadas
        output_discriminator_generated = discriminator(generated_samples)
        
        # Pérdida del generador
        loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
        loss_generator.backward()
        optimizer_generator.step()

    # Mostrar y guardar las pérdidas cada 10 épocas
    if epoch % 10 == 0:
        with open('GAN_log.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, loss_discriminator.item(), loss_generator.item()])
        
        print(f"Epoch {epoch} - Loss Discriminator: {loss_discriminator.item()}, Loss Generator: {loss_generator.item()}")

        # Mostrar algunas secuencias generadas
        with torch.no_grad():
            sample_latent_space = torch.randn((5, latent_dim)).to(device)
            random_lengths = torch.randint(1, max_seq_length + 1, (5,)).to(device)
            generated_samples = generator(sample_latent_space, random_lengths)
            generated_samples = generated_samples.cpu().numpy()

            # Convertir una secuencia generada a rna
            for i, sample in enumerate(generated_samples):
                rna_seq = one_hot_to_rna_with_padding(sample)
                print(f"Secuencia {i + 1} generada en la época {epoch}: {rna_seq}")
