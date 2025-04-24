import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results\seq.csv')

epochs_to_plot = [0,25, 50, 75, 100, 150, 200, 300, 400, 599]

for epoch in epochs_to_plot:
    epoch_df = df[df['epoch'] == epoch]
    
    plt.figure(figsize=(10, 5))
    
    #plt.hist(epoch_df[epoch_df['Type'] == 'Real']['Discriminator Loss'], bins=20, alpha=0.7, 
    #         label='Real Sequences', color='green')
   
    #plt.hist(epoch_df[epoch_df['Type'] == 'Generated']['Discriminator Loss'], bins=20, alpha=0.7, 
    #         label='Generated Sequences', color='blue')
    
    #plt.hist(epoch_df['padding_loss'], bins=20, alpha=0.7, label='Loss Generador', color='blue')
    #plt.hist(epoch_df['loss'], bins=20, alpha=0.7, label='Padding Loss', color='green')
    plt.hist(epoch_df['loss'], bins=20, alpha=0.7, label='Loss', color='green')
    
    plt.xlabel('Loss')
    plt.title(f'Epoch {epoch}')
    plt.legend()
    
    plt.show()