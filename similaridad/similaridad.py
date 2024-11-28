import pandas as pd
from utils import max_similarity


input_file = '../../sincfold-private/data/ArchiveII.csv'
generadas = pd.read_csv('../results/generated_seq_20241101_104535.csv')
resultados =[]
secuencias = generadas['sequence']
epocas = generadas['epoch']

for epoch, row in zip(epocas, secuencias):
    
    if 'X' in row:
        fin = row.index('X')  
        rna_seq = row[:fin]
    else:
        rna_seq = row
    
    score = max_similarity(input_file, rna_seq)

    resultados.append([epoch, rna_seq, score])

df_resultados = pd.DataFrame(resultados, columns=['epoch', 'sequence', 'score'])


df_resultados.to_csv('similaridad_20241101_104535.csv', index=False)
