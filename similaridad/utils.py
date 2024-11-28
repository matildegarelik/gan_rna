import numpy as np
import pandas as pd
from Bio import pairwise2
import multiprocessing as mp

def calcular_medoide(matriz, indices_cluster):
    """
    Calcula el medoide de un cluster dado.
    """
    submatriz = matriz.loc[indices_cluster, indices_cluster]
    distancias = submatriz.sum(axis=1)  # Suma de distancias por fila
    medoide = distancias.idxmin()  # Índice con la menor suma de distancias
    return medoide

def k_medoides(matriz, k=3, max_iter=100):
    """
    Implementación manual de k-medoides.
    """
    secuencias = list(matriz.index)
    if k > len(secuencias):
        medoides = secuencias
    else:
        medoides = np.random.choice(secuencias, size=k, replace=False)  # Elegir medoides iniciales
        for _ in range(max_iter):
            # Asignar cada punto al cluster más cercano
            clusters = {medoide: [] for medoide in medoides}
            for secuencia in secuencias:
                distancias = {medoide: matriz.loc[secuencia, medoide] for medoide in medoides}
                cluster_cercano = min(distancias, key=distancias.get)
                clusters[cluster_cercano].append(secuencia)

            # Recalcular medoides
            nuevos_medoides = []
            for medoide, indices_cluster in clusters.items():
                if indices_cluster:  # Evitar clusters vacíos
                    nuevo_medoide = calcular_medoide(matriz, indices_cluster)
                    nuevos_medoides.append(nuevo_medoide)

            # Verificar si los medoides cambiaron
            if set(nuevos_medoides) == set(medoides):
                break
            medoides = nuevos_medoides

    return medoides

def generar_matrices_por_familia_y_cluster(df, k=3):
    """
    Filtra matrices por familia y realiza clustering k-medoides para cada una.
    """
    familias = set([id.split('_')[0] for id in df.index])  # Identificar familias
    resultados = []

    for familia in familias:
        # Filtrar matriz por familia
        ids_familia = [id for id in df.index if id.startswith(familia)]
        matriz_familia = df.loc[ids_familia, ids_familia]

        # Realizar clustering k-medoides
        medoides = k_medoides(matriz_familia, k=k)
        resultados.extend(medoides)

    return resultados


def calculate_similarity(row_seq, single_seq, match_dic, gap_open, gap_extend):
    score = pairwise2.align.localds(
        single_seq, row_seq, match_dic, gap_open, gap_extend, 
        one_alignment_only=True, score_only=True
    )
    if not isinstance(score, (int, float)):
        score = score[0] if score else 0  # Maneja posibles errores de lista vacía
    max_len = max(len(single_seq), len(row_seq))
    IDscore = score / max_len
    return IDscore

def max_similarity(input_file, single_seq, match=1, mismatch=-2, gap_open=-5, gap_extend=-2):
    simbolos = ["A", "C", "G", "U"]
    
    # crear diccionario de puntuación
    match_dic = {(a, b): match if a == b else mismatch for a in simbolos for b in simbolos}
    
    # dejar solo 3 secuencias por familia para comparar similaridad
    dref_unfiltered = pd.read_csv(input_file)
    dref_unfiltered = dref_unfiltered[dref_unfiltered['len'] <= 200]
    dref_unfiltered['family'] = dref_unfiltered['id'].apply(lambda x: x.split('_')[0])

    
    ids_dref = dref_unfiltered['id'].to_list()  # Extrae los IDs únicos del DataFrame dref
    matriz_distancia = pd.read_hdf('seqsim_f_all.h5')
    matriz_distancia = matriz_distancia.loc[ids_dref, ids_dref]

    medoides = generar_matrices_por_familia_y_cluster(matriz_distancia, k=3)
    dref = dref_unfiltered[dref_unfiltered['id'].isin(medoides)]

    
    # paralelizar calculos de similaridad usando mp
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(
            calculate_similarity,
            [(dref.loc[row].sequence, single_seq, match_dic, gap_open, gap_extend) for row in dref.index]
        )

    
    sims = pd.DataFrame(results, index=dref.index, columns=['similarity_with_single'])
    max_similarity_score = sims['similarity_with_single'].max()
    return max_similarity_score


