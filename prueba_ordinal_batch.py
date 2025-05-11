import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from utils import rna_to_one_hot, one_hot_to_continuous_batch, one_hot_to_rna_with_padding, continuous_to_one_hot_batch
from model import GeneratorCNNOrdinal

# ============ PARÁMETROS ============
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
max_seq_length = 200
batch_size = 32
num_epochs = 5
lr = 0.1
print(f"Usando dispositivo: {device}")
# =====================================

# ============ DATOS ============
df = pd.read_csv('./ArchiveII.csv')
df = df[df['len'] <= max_seq_length]
sequences = df['sequence'].tolist()

# Convertir a one-hot con padding
real_data = np.array([rna_to_one_hot(seq, max_seq_length) for seq in sequences])  # (N, 200, 4)
real_data = torch.tensor(real_data, dtype=torch.float)  # (N, 200, 4)
train_labels = torch.zeros(len(real_data))  # No se usan pero se necesita para el DataLoader
train_set = [(real_data[i], train_labels[i]) for i in range(len(real_data))]
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
# =====================================

# ============ MODELO ============
generator = GeneratorCNNOrdinal(input_dim=1, output_dim=1, max_seq_length=max_seq_length).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(generator.parameters(), lr=lr)
# =====================================

# ============ ENTRENAMIENTO ============
generator.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch, _ in train_loader:
        one_hot_batch = batch.to(device)  # (B, 200, 4)
        cont_batch = one_hot_to_continuous_batch(one_hot_batch, max_seq_length, device)  # (B, 1, 200)

        optimizer.zero_grad()
        output = generator(cont_batch)  # (B, 1, 200)
        loss = criterion(output, cont_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Época {epoch + 1}/{num_epochs} - Loss promedio: {avg_loss:.6f}")
# =====================================

# ============ RESULTADO ==============
generator.eval()
with torch.no_grad():
    for batch, _ in train_loader:
        one_hot_batch = batch.to(device)
        cont_batch = one_hot_to_continuous_batch(one_hot_batch, max_seq_length, device)
        output = generator(cont_batch)  # (B, 1, 200)
        break  # Solo un batch de prueba

output_seq = output.squeeze(1).cpu()  # (B, 200)
one_hot = continuous_to_one_hot_batch(output_seq)  # (B, 200, 4)

# Mostrar una secuencia convertida a RNA
seq = one_hot_to_rna_with_padding(one_hot[0].numpy())
print("Ejemplo de secuencia generada:", seq)

# Mostrar filtros del generador
print("\nPesos finales del generador:")
print(generator.conv.weight.data.squeeze().cpu().numpy())
