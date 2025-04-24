import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np
from utils import rna_to_one_hot, one_hot_to_rna_with_padding


class GeneratorCNN(nn.Module):
    def __init__(self, input_dim=4, output_dim=4, max_seq_length=200):
        super(GeneratorCNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_seq_length = max_seq_length

        self.conv = nn.Conv1d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, groups=input_dim, bias=False)

        #self.conv.weight = nn.Parameter(torch.zeros(input_dim, 1, 3))
        #self.conv.weight.data[:, 0, 1] = 1.0  

    def forward(self, x):
        return self.conv(x) 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

fixed_sequence = "AUGGCUACGUAUGCGAUCGUACGUA"
max_seq_length = 200
input_dim = 4
print(f"Secuencia entrada: {fixed_sequence}")

one_hot_sequence = rna_to_one_hot(fixed_sequence, max_seq_length)
one_hot_tensor = torch.tensor(one_hot_sequence, dtype=torch.float).unsqueeze(0).to(device)
one_hot_tensor = one_hot_tensor.permute(0, 2, 1)

generator = GeneratorCNN(input_dim=4, output_dim=4, max_seq_length=max_seq_length).to(device)

# ====== ENTRENAMIENTO ============
criterion = nn.MSELoss()
optimizer = optim.Adam(generator.parameters(), lr=0.01)
generator.train()
for epoch in range(1000):
    optimizer.zero_grad()
    output = generator(one_hot_tensor)
    loss = criterion(output, one_hot_tensor)
    loss.backward()
    optimizer.step()
    print(f"Época {epoch+1}/10 - Pérdida: {loss.item():.6f}")
# =================================

generator.eval()
with torch.no_grad():
    output = generator(one_hot_tensor)

output = output.permute(0, 2, 1)
output_np = output.squeeze(0).cpu().numpy()

generated_rna_sequence = one_hot_to_rna_with_padding(output_np)
print("Secuencia generada:", generated_rna_sequence)

print("Pesos finales del generador:")
print(generator.conv.weight.data)

print("Identidad:")
v= torch.zeros(input_dim, 1, 3)
v[:, 0, 1] = 1.0  
print(v)

