import torch.nn as nn
import torch

class GeneratorCNN(nn.Module):
    def __init__(self, input_dim=4, output_dim=4, max_seq_length=200):
        super(GeneratorCNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_seq_length = max_seq_length

        # Convolución que mantiene tamaño
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False)
        #self.conv.bias.data.fill_(0.0)
        
        # Peso shape esperado: (output_dim, input_dim, kernel_size)
        self.conv.weight = nn.Parameter(
            torch.zeros(output_dim, input_dim, 3, device='cpu', requires_grad=True)
        )
        self.conv.weight.data[:, :, 1] = 1.0  # filtro tipo identidad desplazada

    def forward(self, x):  # x: (batch_size, input_dim, seq_length)
        return self.conv(x)  # (batch_size, output_dim, seq_length)
