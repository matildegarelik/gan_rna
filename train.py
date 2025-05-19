import csv
import torch
import torch.nn as nn
from torch.optim import Adam
from utils import generate_latent_space_samples, one_hot_to_continuous_batch,one_hot_to_rna_with_padding, continuous_to_rna
from model import padding_loss
import torch.nn.functional as F

def train(generator, discriminator, train_loader, loss_function, optimizer_discriminator, optimizer_generator, num_epochs, device, latent_dim, max_seq_length, mu, log_filename, real_seq_filename, generated_seq_filename, losses_filename):

    real_seq_logfile = open(real_seq_filename, mode='a', newline='')
    real_seq_logger = csv.writer(real_seq_logfile) 
    real_seq_logger.writerow(["epoch", "discriminator_batch", "sequence", "loss"]) 
    
    #losses_logfile = open(losses_filename, mode='a', newline='')
    #losses_logger = csv.writer(losses_logfile) 
    #losses_logger.writerow(["epoch","loss_gen", "padding_loss","mse_loss", "loss_real_disc",'loss_gen_disc','Train Disc/Gen']) # header

    generated_seq_logfile = open(generated_seq_filename, mode='a', newline='')
    generated_seq_logger = csv.writer(generated_seq_logfile) 
    generated_seq_logger.writerow(["epoch", "discriminator_batch", "sequence", "loss"]) 

    train_phase = "discriminator"
    last_discriminator_loss = None
    last_generator_loss = None
    umbral_discriminador = 1.5  # 50% aumento
    umbral_generador = 1.02      # 2% aumento
    ventaja = 5

    discriminator_losses = []
    generator_losses = []

    criterio = nn.MSELoss()
    
    for epoch in range(num_epochs):

        epoch_discriminator_losses = []
        epoch_generator_losses = []

        for n, (real_samples, _) in enumerate(train_loader):
            real_samples = real_samples.to(device)
            real_samples = one_hot_to_continuous_batch(real_samples, max_seq_length, device)
            real_samples_labels = torch.ones((real_samples.size(0), 1)).to(device) 

            optimizer_generator.zero_grad()
            optimizer_discriminator.zero_grad()

            # generar muestras falsas
            # latent_space_samples, random_lengths = generate_latent_space_samples(real_samples.size(0), max_seq_length, device)
            generated_samples = generator(real_samples)
            generated_samples_labels = torch.zeros((real_samples.size(0), 1)).to(device)
            labels = torch.cat((real_samples_labels,generated_samples_labels))
            
            if epoch < ventaja or train_phase == "discriminator":
                # entrenamiento del discriminador
                discriminator.train()
                generator.eval()

                # salida del discriminador para reales y generadas
                output_discriminator = discriminator(torch.cat((real_samples,generated_samples)))

                # loss discriminador
                loss_discriminator = loss_function(output_discriminator, labels)
                epoch_discriminator_losses.append(loss_discriminator.item())

                
                # loss generador
                #padding_loss_value = padding_loss(generated_samples, random_lengths, device)  # (32)
                #generator_loss_value = loss_function(output_discriminator_generated, real_samples_labels)  # (32,1)
                #loss_generator = criterio(generated_samples, real_samples)
                #loss_generator = (generator_loss_value + mu * padding_loss_value + mse_generator_loss).mean()
                #gen_labels = torch.ones((real_samples.size(0), 1)).to(device)  # el generador quiere engañar al discriminador
                
                loss_generator = -1*loss_function(output_discriminator, labels)
                epoch_generator_losses.append(loss_generator.item())
                last_discriminator_loss= loss_discriminator.item()

                loss_discriminator.backward()
                optimizer_discriminator.step()


            elif train_phase == "generator":
            
                discriminator.eval()
                generator.train()

                #latent_space_samples, random_lengths = generate_latent_space_samples(real_samples.size(0), max_seq_length, device)
                generated_samples = generator(real_samples)
                output_discriminator = discriminator(torch.cat((real_samples,generated_samples)))

                # loss generador 
                #padding_loss_value = padding_loss(generated_samples, random_lengths, device)  # (32)
                #generator_loss_value = loss_function(output_discriminator_generated, real_samples_labels)  # (32,1)
                #loss_generator = criterio(generated_samples, real_samples)
                #loss_generator = (generator_loss_value + mu * padding_loss_value + mse_generator_loss).mean()
                #gen_labels = torch.ones((real_samples.size(0), 1)).to(device) # el generador quiere engañar al discriminador
                loss_generator = -1*loss_function(output_discriminator, labels)
                epoch_generator_losses.append(loss_generator.item())

                # loss discriminador
                loss_discriminator = loss_function(output_discriminator, labels)
                epoch_discriminator_losses.append(loss_discriminator.item())

                loss_generator.backward()
                optimizer_generator.step()

            if epoch in (6,99):
                # log losses y sequences
                for i in range(len(generated_samples)):
                    generated_rna_seq = continuous_to_rna(generated_samples[i].cpu().detach().numpy())  
                    generated_seq_logger.writerow([epoch, n, generated_rna_seq, f"{loss_generator.item():.2f}"])                          

                # log secuencias reales
                for i in range(len(generated_samples)):
                    real_rna_seq = continuous_to_rna(real_samples[i].cpu().detach().numpy())
                    real_seq_logger.writerow([epoch, n, real_rna_seq, "-"])

        # Al finalizar la época, decidir cambio de fase
        if epoch > ventaja:
            if train_phase == "discriminator":
                if last_generator_loss is not None and epoch_generator_losses[-1] > last_generator_loss * umbral_generador or last_d<1:
                    print("Cambiando a entrenamiento del generador")
                    train_phase = "generator"
                    
                if epoch_discriminator_losses:
                    last_discriminator_loss = epoch_discriminator_losses[-1]

            elif train_phase == "generator":
                if last_discriminator_loss is not None and epoch_discriminator_losses[-1] > last_discriminator_loss * umbral_discriminador:
                    print("Cambiando a entrenamiento del discriminador")
                    train_phase = "discriminator"
                    
                if epoch_generator_losses:
                    last_generator_loss = epoch_generator_losses[-1]
        elif epoch==ventaja:
            print("Cambiando a entrenamiento del generador")
            train_phase= "generator"

        discriminator_losses.extend(epoch_discriminator_losses)
        generator_losses.extend(epoch_generator_losses)

        # mostrar pérdidas
        last_d = epoch_discriminator_losses[-1] if epoch_discriminator_losses else 0
        last_g = epoch_generator_losses[-1] if epoch_generator_losses else 0
        print(f"Epoch {epoch} - Loss Discriminator: {last_d} | Loss Generator: {last_g} | Training: {train_phase}")

        with open(log_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, f"{loss_discriminator.item():.6f}", f"{loss_generator.item():.6f}", train_phase])
    real_seq_logfile.close()
    #losses_logfile.close()
