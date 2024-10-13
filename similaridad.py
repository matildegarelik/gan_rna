import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import pairwise2
import multiprocessing as mp

def calculate_similarity(row_seq, single_seq, match_dic, gap_open, gap_extend):
    score = pairwise2.align.localds(
        single_seq, row_seq, match_dic, gap_open, gap_extend, 
        one_alignment_only=True, score_only=True
    )
    max_len = max(len(single_seq), len(row_seq))
    IDscore = score / max_len
    return IDscore

def max_similarity(input_file, single_seq, match=1, mismatch=-2, gap_open=-5, gap_extend=-2):
    simbolos = ["A", "C", "G", "U"]
    
    # crear diccionario de puntuación
    match_dic = {(a, b): match if a == b else mismatch for a in simbolos for b in simbolos}
    
    # dejar solo 3 secuencias por familia para comparar similaridad
    dref_unfiltered = pd.read_csv(input_file)
    dref_unfiltered['family'] = dref_unfiltered['id'].apply(lambda x: x.split('_')[0])
    dref = dref_unfiltered.groupby('family').apply(lambda x: x.sample(min(3, len(x)), random_state=42)).reset_index(drop=True)
    
    # paralelizar calculos de similaridad usando mp
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(
            calculate_similarity,
            [(dref.loc[row].sequence, single_seq, match_dic, gap_open, gap_extend) for row in dref.index]
        )

    
    sims = pd.DataFrame(results, index=dref.index, columns=['similarity_with_single'])
    max_similarity_score = sims['similarity_with_single'].max()
    return max_similarity_score


input_file = '../sincfold-private/data/ArchiveII.csv'
generadas = pd.read_csv('results/seq_20241012_112543.csv')
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


df_resultados.to_csv('resultados_similaridad.csv', index=False)
