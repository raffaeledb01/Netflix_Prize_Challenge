import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Reader, Dataset, SVD
from surprise.model_selection import GridSearchCVù
from PCA import PCA

def preprocessing(df):
    df['Rating'] = df['Rating'].astype(float)

    print('Dataset 1 shape: {}'.format(df.shape))
    print('-Dataset examples-')
    print(df)

    df.index = np.arange(0,len(df))
    print('Full dataset shape: {}'.format(df.shape))
    print('-Dataset examples-')
    print(df.iloc[::5000000, :])


    p = df.groupby('Rating')['Rating'].agg(['count'])

    # get movie count
    movie_count = df.isnull().sum()[1]

    # get customer count
    cust_count = df['Cust_Id'].nunique() - movie_count

    # get rating count
    rating_count = df['Cust_Id'].count() - movie_count

    # Leggi il file come testo
    with open('/home/raffaeledb/Statistical Methods/movie_titles.csv', 'r', encoding="ISO-8859-1") as file:
        lines = file.readlines()

    # Funzione per sostituire tutte le virgole eccetto le prime due
    def replace_commas(line):
        # Trova tutte le virgole nella riga
        commas = [m.start() for m in re.finditer(',', line)]
        
        # Se ci sono più di 2 virgole, sostituisci tutte le virgole dopo la seconda
        if len(commas) > 2:
            # Mantieni la parte fino alla seconda virgola e sostituisci le successive
            line = line[:commas[1]+1] + line[commas[1]+1:].replace(',', ';')
        
        return line

    # Applica la funzione a ogni riga
    modified_lines = [replace_commas(line) for line in lines]

    # Salva il testo modificato in un nuovo file (opzionale)
    with open('/home/raffaeledb/Statistical Methods/movie_titles_modified.csv', 'w', encoding="ISO-8859-1") as file:
        file.writelines(modified_lines)

    # Carica il file modificato in un DataFrame
    movie_titles = pd.read_csv('/home/raffaeledb/Statistical Methods/movie_titles_modified.csv', 
                        encoding="ISO-8859-1", 
                        header=None, 
                        names=['Movie_Id', 'Year', 'Name'], 
                        sep=',')  # Usa il nuovo separatore

    print(movie_titles.head(10))

    ax = p.plot(kind = 'barh', legend = False, figsize = (15,10))
    plt.title('Total pool: {:,} Movies, {:,} customers, {:,} ratings given'.format(movie_count, cust_count, rating_count), fontsize=20)
    plt.axis('off')

    for i in range(1,6):
        ax.text(p.iloc[i-1][0]/4, i-1, 'Rating {}: {:.0f}%'.format(i, p.iloc[i-1][0]*100 / p.sum()[0]), color = 'white', weight = 'bold')


    # Get data
    data = df['Rating'].value_counts().sort_index(ascending=True)  # Ordina in ordine crescente

    # Creare un array di stelline corrispondente ai rating
    star_labels = ['★' * int(rating) for rating in data.index]

    # Create bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(star_labels, data.values, color='#db0000')

    # Add text labels on bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, '{:.1f} %'.format(yval / df.shape[0] * 100),
                ha='center', va='bottom', color='#000000')

    # Add title and labels
    plt.title('Distribution Of {} Netflix-Ratings'.format(df.shape[0]))
    plt.xlabel('Rating')
    plt.ylabel('Count')

    # Imposta l'etichetta delle x come stelline gialle
    plt.xticks(ticks=np.arange(len(data.index)), labels=star_labels, fontsize=12, color='gold')

    # Show plot
    plt.tight_layout()
    plt.show()

    # Save the plot as an image file (e.g., PNG)
    plt.savefig("netflix_ratings_distribution_with_stars.png")

    # Ottieni i dati
    data = df.groupby('Movie_Id')['Rating'].count().clip(upper=9999)

    # Crea l'istogramma
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=range(0, 10001, 100), color='#db0000', edgecolor='black')

    # Aggiungi il titolo e le etichette degli assi
    plt.title('Distribution Of Ratings Per Movie (Clipped at 9999)')
    plt.xlabel('Ratings Per Movie')
    plt.ylabel('Count')

    # Mostra il grafico
    plt.grid(True)
    plt.savefig('static_ratings_per_movie.png')  # Salva il grafico come immagine
    plt.close()


    # Ottieni i dati
    data = df.groupby('Cust_Id')['Rating'].count().clip(upper=199)

    # Crea l'istogramma
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=range(0, 201, 2), color='#db0000', edgecolor='black')

    # Aggiungi il titolo e le etichette degli assi
    plt.title('Distribution Of Ratings Per User (Clipped at 199)')
    plt.xlabel('Ratings Per User')
    plt.ylabel('Count')

    # Mostra il grafico
    plt.grid(True)
    plt.savefig('static_ratings_per_user.png')  # Salva il grafico come immagine
    plt.close()

    f = ['count','mean']

    df_movie_summary = df.groupby('Movie_Id')['Rating'].agg(f)
    df_movie_summary.index = df_movie_summary.index.map(int)
    movie_benchmark = round(df_movie_summary['count'].quantile(0.9),0)
    drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index

    print('Movie minimum times of review: {}'.format(movie_benchmark))

    df_cust_summary = df.groupby('Cust_Id')['Rating'].agg(f)
    df_cust_summary.index = df_cust_summary.index.map(int)
    cust_benchmark = round(df_cust_summary['count'].quantile(0.9),0)
    drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index

    print('Customer minimum times of review: {}'.format(cust_benchmark))
    
    # Contare il numero di recensioni per ogni film
    movie_review_counts = df.groupby('Movie_Id').size()

    # Trovare il film con il maggior numero di recensioni
    most_reviewed_movie_id = movie_review_counts.idxmax()
    most_reviewed_movie_count = movie_review_counts.max()

    print(f"Il film con il maggior numero di recensioni ha l'ID {most_reviewed_movie_id} e {most_reviewed_movie_count} recensioni.")

    print('Original Shape: {}'.format(df.shape))
    df = df[~df['Movie_Id'].isin(drop_movie_list)]
    df = df[~df['Cust_Id'].isin(drop_cust_list)]
    print('After Trim Shape: {}'.format(df.shape))
    print('-Data Examples-')
    print(df)

    # Trasformazione del dataset con pivot
    df_pivot = df.pivot(index='Cust_Id', columns='Movie_Id', values='Rating')

    print("Generazione del grafico dei Valori Singolari...")
    try:
        U, sigma, Vt = np.linalg.svd(df_pivot, full_matrices=False)
    except Exception as e:
        print(f"Errore nella creazione del grafico dei valori singolari: {e}")

    plt.figure(figsize=(10, 6))
    plt.plot(sigma, marker='o')
    plt.title('Valori Singolari')
    plt.xlabel('Indice')
    plt.ylabel('Valore Singolare')
    plt.savefig('plots/svd/valori_singolari.png')
    plt.close()


    # Ottieni i dati
    data = movie_titles['Year'].value_counts().sort_index()

    # Crea il grafico
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data.values, color='#db0000', marker='o', markersize=3)

    # Aggiungi il titolo e le etichette degli assi
    plt.title('{} Movies Grouped By Year Of Release'.format(movie_titles.shape[0]))
    plt.xlabel('Release Year')
    plt.ylabel('Movies')

    # Mostra il grafico
    plt.grid(True)
    plt.savefig('static_movies_grouped_by_year.png')  # Salva il grafico come immagine (opzionale)
    plt.close()

    # Ottieni i dati
    data = df['Date'].value_counts()
    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)

    # Crea il grafico
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data.values, color='#db0000', marker='o', markersize=3)

    # Aggiungi il titolo e le etichette degli assi
    plt.title('{} Movie-Ratings Grouped By Day'.format(df.shape[0]))
    plt.xlabel('Date')
    plt.ylabel('Ratings')

    # Mostra il grafico
    plt.grid(True)
    plt.savefig('static_movie_ratings_by_day.png')  # Salva il grafico come immagine (opzionale)
    plt.close()
    
    df_pca = PCA(df, most_reviewed_movie_id)
    
    return df, df_pivot, df_pca, most_reviewed_movie_id