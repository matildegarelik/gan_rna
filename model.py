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
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, max_seq_length),
            nn.Tanh()
        )
        self.max_seq_length = max_seq_length

    def forward(self, x):
        output = self.model(x)
            
        return output


class GeneratorCNN(nn.Module):
    def __init__(self, input_dim, output_dim=1, max_seq_length=196):  # max_seq_length = 196
        super(GeneratorCNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_seq_length = max_seq_length

        # convertir ruido en características iniciales
        #self.linear = nn.Sequential(
        #    nn.Linear(input_dim, 128),
        #    nn.ReLU(),
        #    nn.Linear(128, max_seq_length),  # proyectar al tamaño requerido
        #    nn.ReLU()
        #)

        # bloque convolucional para procesar seq generadas
        self.conv = nn.Sequential(
            nn.Conv1d(1, 20, kernel_size=3, stride=1, padding=1),  # mantener longitud
            nn.ReLU(),
            nn.Conv1d(20, 10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(10, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # salida en el rango [-1, 1]
        )

    def forward(self, x):
        batch_size = x.size(0)

        # Asegurarse de que x tenga el tamaño adecuado
        if x.size(1) != self.max_seq_length:
            x = x.view(batch_size, -1)  # Aplanar cualquier tamaño inicial
            x = x[:, :self.max_seq_length]  # Recortar al tamaño necesario

        # Reorganizar para Conv1d
        x = x.view(batch_size, 1, self.max_seq_length)  # (batch_size, 1, max_seq_length)

        # Bloque convolucional
        x = self.conv(x)  # Salida: (batch_size, 1, max_seq_length)

        # Quitar dimensión del canal para la salida final
        x = x.squeeze(1)  # (batch_size, max_seq_length)

        return x



#calcula la perdida de padding, penalizando las posiciones que deberían estar en padding pero no lo están (no son completamente -1).
#entradas: generated_samples (salidas del generador: batch_size, max_seq_length, 4) y  output_lengths (longitudes esperadas de cada secuencia en el batch)


def padding_loss(generated_samples, output_lengths, device):
    batch_size, max_seq_length = generated_samples.size()
    loss = torch.zeros(batch_size, device=device) # devuelve el loss por secuencia

    for i in range(batch_size):
        length = output_lengths[i]

        # penalización para las posiciones no padding que no son -1
        #loss =  loss + F.mse_loss(generated_samples[i] * mask0, torch.full_like(generated_samples[i], .5) * mask0) + F.mse_loss(generated_samples[i] * mask1, torch.full_like(generated_samples[i], -1) * mask1)
        
        # penalización para las posiciones que no cumplen la longitud esperada.
        generated_samples_clamp = -F.relu(-generated_samples[i,:])
        seq_label = torch.full_like(generated_samples[i], 0, device=device)
        seq_label[length:] = -1 # padding
        
        #seq_loss = F.mse_loss(generated_samples[i, length:], torch.full_like(generated_samples[i, length:], -1))
        seq_loss = F.mse_loss(generated_samples_clamp, seq_label)
        
        loss[i] = seq_loss
    
    return loss