import csv
import torch
import torch.nn as nn
from torch.optim import Adam
from utils import continuous_to_one_hot, generate_latent_space_samples, one_hot_to_rna_with_padding
from model import padding_loss

def train(generator, discriminator, train_loader, loss_function, optimizer_discriminator, optimizer_generator, num_epochs, device, latent_dim, max_seq_length, mu,log_filename, real_seq_filename,generated_seq_filename, losses_filename):
    
    real_seq_logfile = open(real_seq_filename, mode='a', newline='')
    real_seq_logger = csv.writer(real_seq_logfile) 
    real_seq_logger.writerow(["epoch", "discriminator_batch", "sequence", "loss"]) 
    
    losses_logfile = open(losses_filename, mode='a', newline='')
    losses_logger = csv.writer(losses_logfile) 
    losses_logger.writerow(["epoch","loss_gen", "padding_loss", "loss_real_disc",'loss_gen_disc','Train Disc/Gen']) # header

    generated_seq_logfile = open(generated_seq_filename, mode='a', newline='')
    generated_seq_logger = csv.writer(generated_seq_logfile) 
    generated_seq_logger.writerow(["epoch", "discriminator_batch", "sequence", "loss"]) 

    train_gen= False
    train_disc=True
    print_train_gen = False
    print_train_disc = True

    epocas_d = 10      # Número de épocas para entrenar el discriminador
    epocas_g = 70

    for epoch in range(num_epochs):
        if print_train_disc and train_disc:
            print('--------Entrenando solo discriminador-------')
            print_train_disc= False
        elif print_train_gen and train_gen:
            print('--------Entrenando solo generador-------')
            print_train_gen= False
        
        if train_disc:
            traingd='D'
        else:
            traingd='G'
        
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
            
            discriminator.eval()
            generator.train()
            # entrenamiento del generador (5 veces x epoca del discriminador)
            for gen_iter in range(1):
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
                
                # log losses y sequences (si esto es muy lento se puede hacer cada tantas iteraciones)
                for i in range(len(generated_samples)):
                    generated_rna_seq = one_hot_to_rna_with_padding(generated_samples_one_hot[i].cpu().numpy())  
                    generated_seq_logger.writerow([epoch, n, generated_rna_seq, f"{generator_loss_value[i].item():.2f}"])                          
                    losses_logger.writerow([epoch, 
                                         f"{generator_loss_value[i].item()}", f"{padding_loss_value[i].item()}",
                                         f"{loss_discriminator_real[i].item():.2f}",f"{loss_discriminator_generated[i].item():.2f}",
                                         traingd
                                         ])
                    
                # este si es el promedio de la suma
                loss_generator = (generator_loss_value + mu*padding_loss_value).mean()
                    
                if epoch % (epocas_d + epocas_g) < epocas_d and epoch % (epocas_d + epocas_g) != 0:
                    if not train_disc:
                        train_disc=True
                        train_gen= False
                        print_train_gen= True
                    loss_discriminator.backward()
                    optimizer_discriminator.step()
                elif epoch % (epocas_d + epocas_g) >= epocas_d and epoch % (epocas_d + epocas_g) != 0:
                    if not train_gen:
                        train_gen=True
                        train_disc=False
                        print_train_disc=True
                    loss_generator.backward()
                    optimizer_generator.step()
                   
                
            
            #if loss_discriminator.item() > initial_discriminator_loss*1.15 and epoch>10 and train_gen:
            #    train_disc = True
            #    train_gen = False

            #elif epoch>10 and train_disc and loss_generator.item() > initial_generator_loss*1.15:
                

        # mostrar y guardar las pérdidas cada época
        if epoch % 1 == 0:
            with open(log_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch, f"{loss_discriminator.item():.2f}", f"{loss_generator.item():.2f}", traingd])
            
            print(f"Epoch {epoch} - Loss Discriminator: {loss_discriminator.item()}, Loss Generator: {loss_generator.item()} - Training: {traingd}")
        
    real_seq_logfile.close()
    losses_logfile.close()
    