import csv
import torch
import torch.nn as nn
from torch.optim import Adam
from utils import continuous_to_one_hot, generate_latent_space_samples, one_hot_to_rna_with_padding
from model import padding_loss

def train(generator, discriminator, train_loader, loss_function, optimizer_discriminator, optimizer_generator, num_epochs, device, latent_dim, max_seq_length, mu,log_filename, real_seq_filename, gen_seq_filename):
    
    real_seq_logfile = open(real_seq_filename, mode='a', newline='')
    real_seq_logger = csv.writer(real_seq_logfile) 
    real_seq_logger.writerow(["epoch", "discriminator_batch", "sequence", "loss"]) 
    
    gen_seq_logfile = open(gen_seq_filename, mode='a', newline='')
    gen_seq_logger = csv.writer(gen_seq_logfile) 
    gen_seq_logger.writerow(["epoch", "discriminator_batch", "generator_batch", "sequence", "loss", "padding_loss", "gen_len"]) # header
                                
    for epoch in range(num_epochs):
        
        for n, (real_samples, _) in enumerate(train_loader):
            real_samples = real_samples.to(device)
            real_samples_labels = torch.ones((real_samples.size(0), 1)).to(device) * 0.9  # Label smoothing

            # generar muestras falsas
            latent_space_samples, random_lengths = generate_latent_space_samples(real_samples.size(0), max_seq_length, device)
            generated_samples = generator(latent_space_samples)
            generated_samples_one_hot = continuous_to_one_hot(generated_samples).to(device)

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

            # log secuencias reales
            for i in range(len(generated_samples)):
                real_rna_seq = one_hot_to_rna_with_padding(real_samples[i].cpu().numpy())
                real_seq_logger.writerow([epoch, n, real_rna_seq, f"{loss_discriminator_real[i].item():.2f}"])
                
            # Entrenamiento del discriminador combinando las pérdidas reales y generadas 
            # ( ojo que el promedio de la suma no es igual al promedio de la concatenación)
            loss_discriminator = torch.cat((loss_discriminator_real, loss_discriminator_generated)).mean()
            loss_discriminator.backward()
            optimizer_discriminator.step()
                     
            discriminator.eval()
            generator.train()
            # entrenamiento del generador (5 veces x epoca del discriminador)
            for gen_iter in range(5):
                discriminator.zero_grad()
                generator.zero_grad()
                latent_space_samples, random_lengths = generate_latent_space_samples(real_samples.size(0), max_seq_length, device)
                generated_samples = generator(latent_space_samples)
                generated_samples_one_hot = continuous_to_one_hot(generated_samples).to(device)

                # salida del discriminador para las muestras generadas
                output_discriminator_generated = discriminator(generated_samples_one_hot)

                #loss_generator = loss_function(output_discriminator_generated, real_samples_labels) + mu * padding_loss_value
                padding_loss_value = padding_loss(generated_samples, random_lengths, device)
                generator_loss_value = loss_function(output_discriminator_generated, real_samples_labels) 
                
                # log losses and sequences (si esto es muy lento se puede hacer cada tantas iteraciones)
                for i in range(len(generated_samples)):
                    generated_rna_seq = one_hot_to_rna_with_padding(generated_samples_one_hot[i].cpu().numpy())                            
                    gen_seq_logger.writerow([epoch, n, gen_iter, generated_rna_seq, 
                                         f"{generator_loss_value[i].item()}", f"{padding_loss_value[i].item()}", random_lengths[i].item()])
                    
                # este si es el promedio de la suma
                loss_generator = (generator_loss_value + mu*padding_loss_value).mean()
                loss_generator.backward()
                optimizer_generator.step()
                
        # mostrar y guardar las pérdidas cada época
        if epoch % 1 == 0:
            with open(log_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch, f"{loss_discriminator.item():.2f}", f"{loss_generator.item():.2f}" ])
            
            print(f"Epoch {epoch} - Loss Discriminator: {loss_discriminator.item()}, Loss Generator: {loss_generator.item()}")
    
    real_seq_logfile.close()
    gen_seq_logfile.close()
    