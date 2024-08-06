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
            nn.Linear(input_dim+1, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, max_seq_length * output_dim)
        )
        self.max_seq_length = max_seq_length
        self.output_dim = output_dim

    def forward(self, x, lengths):
        lengths = lengths.unsqueeze(1).float() / self.max_seq_length 
        x = torch.cat([x, lengths], dim=1)
        output = self.model(x)
        output = output.view(-1, self.max_seq_length, self.output_dim)
        max_indices = torch.argmax(output, dim=-1)
        one_hot_output = F.one_hot(max_indices, num_classes=self.output_dim).float()
        return one_hot_output
