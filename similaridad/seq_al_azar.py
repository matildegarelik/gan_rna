import random
import pandas as pd
from utils import max_similarity


input_file = '../../sincfold-private/data/ArchiveII.csv'
resultados =[]

# se generan 836 secuencias por época de hasta 100 de longitud final
secuencias=[]
for i in range(0,837):
    longitud = random.randint(1,200)
    secuencia = ""
    for n in range (0, longitud):
        f = random.random()
        if f<0.25:
            secuencia+="A"
        elif f<0.5:
            secuencia+="C"
        elif f<0.75:
            secuencia+="G"
        else:
            secuencia+="U"
    #for n in range (0,100-longitud):
    #    secuencia+="X"
    secuencias.append(secuencia)

    score = max_similarity(input_file, secuencia)
    resultados.append([0, secuencia, score])

df_resultados = pd.DataFrame(resultados, columns=['epoch', 'sequence', 'score'])


df_resultados.to_csv('similaridad_aleatorias_l200.csv', index=False)
