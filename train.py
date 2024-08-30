import csv
import torch
import torch.nn as nn
from torch.optim import Adam
from utils import continuous_to_one_hot, generate_latent_space_samples, one_hot_to_rna_with_padding
from model import padding_loss

def train(generator, discriminator, train_loader, loss_function, optimizer_discriminator, optimizer_generator, num_epochs, device, latent_dim, max_seq_length, mu,log_filename, sequences_filename, gen_filename):
    for epoch in range(num_epochs):
        all_real_samples = []
        all_generated_samples = []
        all_loss_discriminator_real = []
        all_loss_discriminator_generated = []
        all_loss_individual_generator = []
        all_padding_loss_generator = []
        all_all_generated_samples = []

        for n, (real_samples, _) in enumerate(train_loader):
            real_samples = real_samples.to(device)
            real_samples_labels = torch.ones((real_samples.size(0), 1)).to(device) * 0.9  # Label smoothing

            # generar muestras falsas
            latent_space_samples, random_lengths = generate_latent_space_samples(real_samples.size(0), max_seq_length, device)
            generated_samples = generator(latent_space_samples)
            generated_samples_one_hot = continuous_to_one_hot(generated_samples)

            generated_samples_labels = torch.zeros((real_samples.size(0), 1)).to(device)

            # entrenamiento del discriminador
            discriminator.train()
            generator.eval()
            discriminator.zero_grad()
            generator.zero_grad()
            
            # Salida del discriminador para las muestras reales y generadas por separado
            output_discriminator_real = discriminator(real_samples)
            output_discriminator_generated = discriminator(generated_samples_one_hot)

            # Pérdida del discriminador para las muestras reales y generadas
            loss_discriminator_real = loss_function(output_discriminator_real, real_samples_labels)
            loss_discriminator_generated = loss_function(output_discriminator_generated, generated_samples_labels)

            # Entrenamiento del discriminador combinando las pérdidas reales y generadas
            loss_discriminator = (loss_discriminator_real + loss_discriminator_generated).mean()
            loss_discriminator.backward()
            optimizer_discriminator.step()

            discriminator.eval()
            generator.train()
            # entrenamiento del generador (5 veces x epoca del discriminador)
            for _ in range(50):
                discriminator.zero_grad()
                generator.zero_grad()
                latent_space_samples, random_lengths = generate_latent_space_samples(real_samples.size(0), max_seq_length, device)
                generated_samples = generator(latent_space_samples)
                generated_samples_one_hot = continuous_to_one_hot(generated_samples)

                # salida del discriminador para las muestras generadas
                output_discriminator_generated = discriminator(generated_samples_one_hot)

                # pérdida del generador
                for i,s in enumerate(generated_samples):
                    padding_loss_value = padding_loss(s.unsqueeze(0), [random_lengths[i]], device)
                    all_padding_loss_generator.append(padding_loss_value.detach().cpu().numpy())
                #loss_generator = loss_function(output_discriminator_generated, real_samples_labels) + mu * padding_loss_value
                padding_loss_value = padding_loss(generated_samples, random_lengths, device)

                loss_generator_individual = loss_function(output_discriminator_generated, real_samples_labels) 
                padding_loss_individual = mu * padding_loss_value
                loss_generator = (loss_generator_individual + padding_loss_individual).mean()
                loss_generator.backward()
                optimizer_generator.step()
                
                all_loss_individual_generator.append(loss_generator_individual.detach().cpu().numpy())
                all_all_generated_samples.append(generated_samples_one_hot.cpu().numpy())

            all_real_samples.append(real_samples.cpu().numpy())
            all_generated_samples.append(generated_samples_one_hot.cpu().numpy())
            all_loss_discriminator_real.append(loss_discriminator_real.detach().cpu().numpy())
            all_loss_discriminator_generated.append(loss_discriminator_generated.detach().cpu().numpy())

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
                    for i in range(len(all_real_samples)):
                        for j in range(all_real_samples[i].shape[0]):
                            real_rna_seq = one_hot_to_rna_with_padding(all_real_samples[i][j])
                            generated_rna_seq = one_hot_to_rna_with_padding(all_generated_samples[i][j])
                            
                            # Guardar la secuencia real y la pérdida asociada
                            seq_writer.writerow([epoch, real_rna_seq, "Real", all_loss_discriminator_real[i][j].item()])
                            
                            # Guardar la secuencia generada y la pérdida asociada
                            seq_writer.writerow([epoch, generated_rna_seq, "Generated", all_loss_discriminator_generated[i][j].item()])
            
            with open(gen_filename, mode='a', newline='') as gen_seq_file:
                gen_seq_writer = csv.writer(gen_seq_file)
                with torch.no_grad():
                    for i in range(len(all_generated_samples)):
                        for j in range(all_generated_samples[i].shape[0]):
                            generated_rna_seq = one_hot_to_rna_with_padding(all_all_generated_samples[i][j])
                            
                            individual_loss = float(all_loss_individual_generator[i][j])
                            padding_loss_g = float(all_padding_loss_generator[i])
                            
                            gen_seq_writer.writerow([epoch, generated_rna_seq, individual_loss, padding_loss_g])