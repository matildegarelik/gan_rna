import torch
import csv
from utils import one_hot_to_rna_with_padding

def train_gan(generator, discriminator, train_loader, loss_function, optimizer_discriminator, optimizer_generator, num_epochs, device, latent_dim, max_seq_length):
    for epoch in range(num_epochs):
        for n, (real_samples, _) in enumerate(train_loader):
            real_samples = real_samples.to(device)
            real_samples_labels = torch.ones((real_samples.size(0), 1)).to(device) * 0.9  # Label smoothing

            latent_space_samples = torch.randn((real_samples.size(0), latent_dim)).to(device)
            random_lengths = torch.randint(1, max_seq_length + 1, (real_samples.size(0),)).to(device)
            generated_samples = generator(latent_space_samples, random_lengths)

            generated_samples_labels = torch.zeros((real_samples.size(0), 1)).to(device)
            
            all_samples = torch.cat((real_samples, generated_samples))
            all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
            loss_discriminator = loss_function(output_discriminator, all_samples_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()

        for n, (real_samples, _) in enumerate(train_loader):
            real_samples = real_samples.to(device)
            real_samples_labels = torch.ones((real_samples.size(0), 1)).to(device)

            latent_space_samples = torch.randn((real_samples.size(0), latent_dim)).to(device)
            random_lengths = torch.randint(1, max_seq_length + 1, (real_samples.size(0),)).to(device)
            generated_samples = generator(latent_space_samples, random_lengths)
            
            output_discriminator_generated = discriminator(generated_samples)
            
            loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
            loss_generator.backward()
            optimizer_generator.step()

        if epoch % 10 == 0:
            with open('GAN_log.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch, loss_discriminator.item(), loss_generator.item()])
            
            print(f"Epoch {epoch} - Loss Discriminator: {loss_discriminator.item()}, Loss Generator: {loss_generator.item()}")

            with torch.no_grad():
                sample_latent_space = torch.randn((5, latent_dim)).to(device)
                random_lengths = torch.randint(1, max_seq_length + 1, (5,)).to(device)
                generated_samples = generator(sample_latent_space, random_lengths)
                generated_samples = generated_samples.cpu().numpy()

                for i, sample in enumerate(generated_samples):
                    rna_seq = one_hot_to_rna_with_padding(sample)
                    print(f"Secuencia {i + 1} generada en la época {epoch}: {rna_seq}")
