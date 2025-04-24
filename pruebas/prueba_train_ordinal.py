import csv
import torch
import torch.nn as nn
from torch.optim import Adam
from utils import generate_latent_space_samples, one_hot_to_continuous_batch
from model import padding_loss
import torch.nn.functional as F

def train(generator, discriminator, train_loader, loss_function, optimizer_discriminator, optimizer_generator, num_epochs, device, latent_dim, max_seq_length, mu,log_filename, real_seq_filename,generated_seq_filename, losses_filename):

    for epoch in range(num_epochs):
        
        for n, (real_samples, _) in enumerate(train_loader):
            real_samples = real_samples.to(device)
            real_samples = one_hot_to_continuous_batch(real_samples,max_seq_length,device)
            real_samples_labels = torch.ones((real_samples.size(0), 1)).to(device) * 0.9  # Label smoothing

            # generar muestras falsas
            latent_space_samples, random_lengths = generate_latent_space_samples(real_samples.size(0), max_seq_length, device)
            generated_samples = generator(latent_space_samples)
            generated_samples_labels = torch.zeros((real_samples.size(0), 1)).to(device)

            # entrenamiento del discriminador
            discriminator.train()
            generator.eval()
            discriminator.zero_grad()
            generator.zero_grad()
            
            # Salida del discriminador para las muestras reales y generadas por separado
            output_discriminator_real = discriminator(real_samples)
            output_discriminator_generated = discriminator(generated_samples)


            # Pérdida del discriminador para las muestras reales y generadas
            loss_discriminator_real = loss_function(output_discriminator_real, real_samples_labels)
            loss_discriminator_generated = loss_function(output_discriminator_generated, generated_samples_labels)
                
            # Entrenamiento del discriminador combinando las pérdidas reales y generadas 
            loss_discriminator = torch.cat((loss_discriminator_real, loss_discriminator_generated)).mean()
            
            loss_discriminator.backward()
            optimizer_discriminator.step()
        

            discriminator.eval()
            generator.train()

            # entrenamiento del generador
            discriminator.zero_grad()
            generator.zero_grad()
            latent_space_samples, random_lengths = generate_latent_space_samples(real_samples.size(0), max_seq_length, device)
            generated_samples = generator(latent_space_samples)

            # salida del discriminador para las muestras generadas
            output_discriminator_generated = discriminator(generated_samples)

            padding_loss_value = padding_loss(generated_samples, random_lengths, device) #(32)
            generator_loss_value = loss_function(output_discriminator_generated, real_samples_labels) #(32,1)
            mse_generator_loss = torch.mean((generated_samples - latent_space_samples) ** 2, dim=1)  # (32, 200)
            mse_generator_loss = mse_generator_loss.mean(dim=1)  # (32)
            loss_generator = (generator_loss_value + mu*padding_loss_value + mse_generator_loss).mean()

            loss_generator.backward()
            optimizer_generator.step()


        # mostrar y guardar las pérdidas cada época
        if epoch % 1 == 0:
            print(f"Epoch {epoch} - Loss Discriminator: {loss_discriminator.item()}, Loss Generator: {loss_generator.item()} - Training: GyD")
