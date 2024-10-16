import torch
from torch import nn
import torch.nn.functional as F

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
        x = x.view(-1, self.seq_length * self.input_dim)
        padding_mask = (x != -1).float()
        x = x * padding_mask
        return self.model(x)

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, max_seq_length):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, max_seq_length * output_dim)
        )
        self.max_seq_length = max_seq_length
        self.output_dim = output_dim

    def forward(self, x):
        output = self.model(x)
        output = output.view(-1, self.max_seq_length, self.output_dim)

        max_indices = torch.argmax(output, dim=-1)  # Encuentra los índices con los valores máximos para cada posición en la secuencia.
        one_hot_output = F.one_hot(max_indices, num_classes=self.output_dim).float()
        return one_hot_output
    


#calcula la perdida de padding, penalizando las posiciones que deberían estar en padding pero no lo están (no son completamente -1).
#entradas: generated_samples (salidas del generador: batch_size, max_seq_length, 4) y  output_lengths (longitudes esperadas de cada secuencia en el batch)

def padding_loss(generated_samples, output_lengths,device):
    batch_size, max_seq_length, _ = generated_samples.size()
    loss = 0

    for i in range(batch_size):
        length = output_lengths[i]

        # crear una máscara donde las posiciones que deberían estar en padding sean 1
        mask = (torch.arange(max_seq_length).to(device) >= length).float()
        mask = mask.unsqueeze(1).expand_as(generated_samples[i])  # expandir a (max_seq_length, 4)
        
        #print(length)
        #print(mask)

        # penalización para las posiciones no padding que no son [-1, -1, -1, -1]
        loss += F.binary_cross_entropy_with_logits(generated_samples[i] * mask, torch.full_like(generated_samples[i], -1) * mask)
    
    return loss / batch_size
