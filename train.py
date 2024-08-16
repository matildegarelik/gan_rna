import csv
import torch
import torch.nn as nn
from torch.optim import Adam
from utils import continuous_to_one_hot, generate_latent_space_samples, one_hot_to_rna_with_padding
from model import padding_loss

def train(generator, discriminator, train_loader, loss_function, optimizer_discriminator, optimizer_generator, num_epochs, device, latent_dim, max_seq_length, mu,log_filename, sequences_filename):
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

        # entrenamiento del generador (5 veces x epoca del discriminador)
            for _ in range(5):
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
        if epoch % 1 == 0:
            with open(log_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch, loss_discriminator.item(), loss_generator.item()])
            
            print(f"Epoch {epoch} - Loss Discriminator: {loss_discriminator.item()}, Loss Generator: {loss_generator.item()}")

            # Guardar secuencias reales y generadas en un CSV
            with open(sequences_filename, mode='a', newline='') as seq_file:
                seq_writer = csv.writer(seq_file)
                
                with torch.no_grad():
                    latent_space_samples, _ = generate_latent_space_samples(real_samples.size(0), max_seq_length, device)
                    generated_samples = generator(latent_space_samples)
                    generated_samples_one_hot = continuous_to_one_hot(generated_samples).to(device)
                    generated_samples_one_hot = generated_samples_one_hot.cpu().numpy()

                    real_samples_one_hot = real_samples.cpu().numpy()

                    for i in range(real_samples.size(0)):
                        real_rna_seq = one_hot_to_rna_with_padding(real_samples_one_hot[i])
                        generated_rna_seq = one_hot_to_rna_with_padding(generated_samples_one_hot[i])
                        seq_writer.writerow([epoch, real_rna_seq, generated_rna_seq, loss_discriminator.item()])
                        print(f"Secuencia {i + 1} generada en la época {epoch}: {generated_rna_seq}")
            

