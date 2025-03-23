import torch
import numpy as np
from model import GeneratorCNN
from utils import rna_to_one_hot, one_hot_to_rna_with_padding

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
input_dim = 4
latent_dim = 200 
max_seq_length = 200 

generator = GeneratorCNN(input_dim=latent_dim, output_dim=input_dim, max_seq_length=max_seq_length).to(device)
generator.eval()

fixed_sequence = "AUGGCUACGUAUGCGAUCGUACGUA" 
print(f"Secuencia entrada: {fixed_sequence}")
one_hot_sequence = rna_to_one_hot(fixed_sequence, max_seq_length)

#print(f"{one_hot_sequence.shape}")  # (max_seq_length, 4)

# padding
if one_hot_sequence.shape[0] < max_seq_length:
    padding = max_seq_length - one_hot_sequence.shape[0]
    one_hot_sequence = np.pad(one_hot_sequence, ((0, padding), (0, 0)), mode='constant', constant_values=-1)

# convertir a tensor y agregar dimensión batch: el tensor original es de forma (max_seq_length, 4)
one_hot_tensor = torch.tensor(one_hot_sequence, dtype=torch.float).unsqueeze(0).to(device) 
#print(f"{one_hot_tensor.shape}")  # (1, max_seq_length, 4)

# reorganizar el tensor para la entrada de la red convolucional: el generador espera (batch_size, 1, max_seq_length)
one_hot_tensor = one_hot_tensor.permute(0, 2, 1)  # (batch_size, input_dim, seq_length)
one_hot_tensor = one_hot_tensor.contiguous()

with torch.no_grad():
    output = generator(one_hot_tensor) 

#print(output.shape)  
#print(output)

generated_rna_sequence = one_hot_to_rna_with_padding(output.cpu().numpy().squeeze())
print("Secuencia generada:", generated_rna_sequence)
