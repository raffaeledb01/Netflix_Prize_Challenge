import pandas as pd
from preprocessing import preprocessing
from SVD import SVD
from NMF import NMF
from NN import NN_task_1, NN_task_2
from autoencoder import autoencoder


if __name__ == '__main__':

    
    print('Inizio esecuzione')
    df = pd.read_csv('../dataset/combined_data_1.txt', header = None, names = ['Cust_Id', 'Rating', 'Date'], usecols = [0,1,2])
    
    # Preprocessing 
    df_preproc, df_pivot, df_pca, specific_movie_id = preprocessing(df)
    print('Preprocessing completed successfully')
    
    # SVD
    SVD(df_preproc)
    print('SVD algorithm execution completed successfully')
    
    # NMF
    NMF(df_preproc)
    print('NMF algorithm execution completed successfully')
    
    # Neural Network Task 1
    NN_task_1(df_preproc)
    print('Neural Network Task 1 execution completed successfully')
    
    # Neural Network Task 2
    NN_task_2(df_pca, specific_movie_id)
    print('Neural Network Task 2 execution completed successfully')
    
    # Autoencoder
    autoencoder(df_preproc, df_pivot)
    print('Autoencoder execution completed successfully')
    