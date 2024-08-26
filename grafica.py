import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results\seq_20240825_184511.csv')

epochs_to_plot = [0, 25, 50, 75, 100, 150, 200, 300, 400, 600]

for epoch in epochs_to_plot:
    epoch_df = df[df['Epoch'] == epoch]
    
    plt.figure(figsize=(10, 5))
    plt.hist(epoch_df['Discriminator Loss'], bins=20, alpha=0.7, label='Loss discriminador', color='blue')
    
    plt.xlabel('Loss')
    plt.title(f'Epoch {epoch}')
    plt.legend()
    
    plt.show()
