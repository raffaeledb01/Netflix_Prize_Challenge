import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA




def PCA(df, most_reviewed_movie_id):
    df_pivot = df.pivot(index='Cust_Id', columns='Movie_Id', values='Rating')
    # Separare la colonna specific_movie_id
    specific_movie_id = most_reviewed_movie_id  # Sostituisci con il tuo Movie_Id specifico
    # Crea una colonna per tenere traccia se inizialmente c'era NaN
    df_pivot['was_nan'] = df_pivot[specific_movie_id].isna()

    # Funzione per riempire i valori NaN con la media della riga
    def fill_na_with_row_mean(df):
        # Calcola la media per ogni riga ignorando i NaN
        row_means = df.mean(axis=1)
        # Sostituisce i NaN con la media della rispettiva riga
        return df.apply(lambda row: row.fillna(row.mean()), axis=1)

    # Applica la funzione al DataFrame per riempire i NaN con la media della riga
    df_pivot_filled = fill_na_with_row_mean(df_pivot.drop(columns=['was_nan']))

    # Ripristina l'indice e aggiungi la colonna 'was_nan' di nuovo
    df_pivot_filled['was_nan'] = df_pivot['was_nan'].values
    df_pivot_filled.reset_index(drop=True, inplace=True)
    
    # Separare la colonna specific_movie_id
    specific_movie_id = 1905  # Sostituisci con il tuo Movie_Id specifico
    column_to_keep = df_pivot_filled[specific_movie_id]
    df_for_pca = df_pivot_filled.drop(columns=[specific_movie_id, 'was_nan'])

    # PCA completa per ottenere tutti gli autovalori
    pca_full = PCA()
    pca_full.fit(df_for_pca)
    
    # Grafico della varianza spiegata cumulativa
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker='o')
    plt.title('Varianza Spiegata Cumulativa per Numero di Componenti')
    plt.xlabel('Numero di Componenti')
    plt.ylabel('Varianza Spiegata Cumulativa')
    plt.grid(True)
    plt.savefig('varianza_spiegata_cumulativa.png')
    
    # Grafico degli autovalori (varianza spiegata per componente)
    plt.figure(figsize=(8, 5))
    plt.plot(pca_full.explained_variance_, marker='o')
    plt.title('Autovalori (Varianza Spiegata) per Numero di Componenti')
    plt.xlabel('Numero di Componenti')
    plt.ylabel('Autovalore (Varianza Spiegata)')
    plt.grid(True)
    plt.savefig('varianza_spiegata_per_componente.png')
    
    # Riduzione delle dimensioni con PCA
    pca = PCA(n_components=120)
    df_pca = pca.fit_transform(df_for_pca)
    print(df_pca)
    
    # Converti il risultato PCA in DataFrame e aggiungi la colonna 'was_nan'
    df_pca_df = pd.DataFrame(df_pca)
    df_pca_df['was_nan'] = df_pivot_filled['was_nan'].values
    df_pca_df[specific_movie_id] = column_to_keep.values
    
    
    return df_pca_df