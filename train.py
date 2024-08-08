import csv
import torch
import torch.nn as nn
from torch.optim import Adam
from utils import continuous_to_one_hot, generate_latent_space_samples, one_hot_to_rna_with_padding
from model import padding_loss

def train(generator, discriminator, train_loader, loss_function, optimizer_discriminator, optimizer_generator, num_epochs, device, latent_dim, max_seq_length, mu):
    for epoch in range(num_epochs):
        for n, (real_samples, _) in enumerate(train_loader):
            real_samples = real_samples.to(device)
            real_samples_labels = torch.ones((real_samples.size(0), 1)).to(device) * 0.9  # Label smoothing

            # generar muestras falsas
            latent_space_samples, random_lengths = generate_latent_space_samples(real_samples.size(0), max_seq_length, device)
            generated_samples = generator(latent_space_samples)
            generated_samples_one_hot = continuous_to_one_hot(generated_samples)

            generated_samples_labels = torch.zeros((real_samples.size(0), 1)).to(device)
            
            # mezclar datos reales y generados
            all_samples = torch.cat((real_samples, generated_samples_one_hot))
            all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

            # entrenamiento del discriminador
            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
            loss_discriminator = loss_function(output_discriminator, all_samples_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()

        # entrenamiento del generador
        for n, (real_samples, _) in enumerate(train_loader):
            real_samples = real_samples.to(device)
            real_samples_labels = torch.ones((real_samples.size(0), 1)).to(device)

            # datos para entrenar el generador
            latent_space_samples, random_lengths = generate_latent_space_samples(real_samples.size(0), max_seq_length, device)
            generated_samples = generator(latent_space_samples)
            generated_samples_one_hot = continuous_to_one_hot(generated_samples)

            # salida del discriminador para las muestras generadas
            output_discriminator_generated = discriminator(generated_samples_one_hot)

            # pérdida del generador
            padding_loss_value = padding_loss(generated_samples, random_lengths, device)
            loss_generator = loss_function(output_discriminator_generated, real_samples_labels) + mu * padding_loss_value
            loss_generator.backward()
            optimizer_generator.step()

        # mostrar y guardar las pérdidas cada 10 épocas
        if epoch % 10 == 0:
            with open('GAN_log.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch, loss_discriminator.item(), loss_generator.item()])
            
            print(f"Epoch {epoch} - Loss Discriminator: {loss_discriminator.item()}, Loss Generator: {loss_generator.item()}")

            # mostrar algunas secuencias generadas
            with torch.no_grad():
                latent_space_samples, _ = generate_latent_space_samples(real_samples.size(0), max_seq_length, device)
                generated_samples = generator(latent_space_samples)
                generated_samples_one_hot = continuous_to_one_hot(generated_samples)

                generated_samples_one_hot = generated_samples_one_hot.cpu().numpy()

                for i, sample in enumerate(generated_samples_one_hot):
                    rna_seq = one_hot_to_rna_with_padding(sample)
                    print(f"Secuencia {i + 1} generada en la época {epoch}: {rna_seq}")
                    #print(f"Entrada del generador: {latent_space_samples[i]}")
