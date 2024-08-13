import torch
from utils import one_hot_to_continuous

def pretrain_generator_as_autoencoder(generator, real_samples, optimizer, loss_function, num_epochs, device, max_seq_length, noise_levels):
    generator.train()

    for epoch in range(num_epochs):
        for real_sample in real_samples:
            real_sample = real_sample.to(device)
            for noise_level in noise_levels:
                noisy_input = real_sample + torch.randn_like(real_sample) * noise_level
                noisy_input = torch.clamp(noisy_input, min=-1, max=1)  #  que los valores esten entre -1 y 1
                #print(one_hot_to_continuous(noisy_input))
                
                optimizer.zero_grad()
                output = generator(one_hot_to_continuous(noisy_input))
                output = output.view(-1, max_seq_length)
                
                continuous_real_sample = one_hot_to_continuous(real_sample).view(-1)
                loss = loss_function(output.view(-1), continuous_real_sample)
                loss.backward()
                optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

