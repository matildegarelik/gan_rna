import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from utils import rna_to_one_hot, one_hot_to_continuous, continuous_to_one_hot_batch, one_hot_to_rna_with_padding

class GeneratorCNN(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, max_seq_length=200):
        super(GeneratorCNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_seq_length = max_seq_length

        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False)
        
        #self.conv.weight.data=nn.Parameter(torch.zeros(input_dim, 1, 3))
        #self.conv.weight.data[0, 0, 1] = 1.0 

    def forward(self, x):
        return self.conv(x)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

fixed_sequence = "AUGGCUACGUAUGCGAUCGUACGUA"
max_seq_length = 200
print(f"Secuencia entrada: {fixed_sequence}")

one_hot_sequence = rna_to_one_hot(fixed_sequence, max_seq_length)  # (200, 4)
one_hot_tensor = torch.tensor(one_hot_sequence, dtype=torch.float)
continuous_tensor = one_hot_to_continuous(one_hot_tensor)  # (200,)
continuous_tensor = continuous_tensor.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 200)

generator = GeneratorCNN(input_dim=1, output_dim=1, max_seq_length=max_seq_length).to(device)

# ====== ENTRENAMIENTO ============
criterion = nn.MSELoss()
optimizer = optim.Adam(generator.parameters(), lr=0.1)
generator.train()
for epoch in range(100):
    optimizer.zero_grad()
    output = generator(continuous_tensor)
    loss = criterion(output, continuous_tensor)
    loss.backward()
    optimizer.step()
    #print(f"Época {epoch+1}/10 - Pérdida: {loss.item():.6f}")
# =================================

# ======  RESULTADOS =======
generator.eval()
with torch.no_grad():
    output = generator(continuous_tensor)

output_seq = output.squeeze().cpu().numpy()  # (200,)
output_seq_tensor = torch.tensor(output_seq).unsqueeze(0)  # (1, 200)
one_hot = continuous_to_one_hot_batch(output_seq_tensor)
seq = one_hot_to_rna_with_padding(one_hot.squeeze(0).numpy())

print("Secuencia generada:", seq)

# Mostrar pesos finales
print("\nPesos finales del generador:")
print(generator.conv.weight.data.squeeze().cpu().numpy())
