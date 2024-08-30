import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results\seq_20240826_094616.csv')

epochs_to_plot = [0,25, 50, 75, 100, 150, 200, 300, 400, 599]

for epoch in epochs_to_plot:
    epoch_df = df[df['Epoch'] == epoch]
    
    plt.figure(figsize=(10, 5))
    
    plt.hist(epoch_df[epoch_df['Type'] == 'Real']['Discriminator Loss'], bins=20, alpha=0.7, 
             label='Real Sequences', color='green')
   
    plt.hist(epoch_df[epoch_df['Type'] == 'Generated']['Discriminator Loss'], bins=20, alpha=0.7, 
             label='Generated Sequences', color='blue')
    
    plt.xlabel('Loss')
    plt.title(f'Epoch {epoch}')
    plt.legend()
    
    plt.show()
