import pandas as pd
import numpy as np
import math
import re
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.decomposition import PCA
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers


def NN_task_1(df):
    
    
    # Codifica le colonne 'Cust_Id' e 'Movie_Id' come interi
    user_encoder = LabelEncoder()
    df['Cust_Id'] = user_encoder.fit_transform(df['Cust_Id'])

    movie_encoder = LabelEncoder()
    df['Movie_Id'] = movie_encoder.fit_transform(df['Movie_Id'])

    # Divisione in train e test set
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    # Prepara i dati per il modello
    x_train = [train['Cust_Id'].values, train['Movie_Id'].values]
    y_train = train['Rating'].values

    x_test = [test['Cust_Id'].values, test['Movie_Id'].values]
    y_test = test['Rating'].values
    
    # Parametri del modello
    num_users = df['Cust_Id'].nunique()
    num_movies = df['Movie_Id'].nunique()
    embedding_size = 50  # Dimensione degli embedding

    # Input per utenti e film
    user_input = layers.Input(shape=(1,), name='user_input')
    movie_input = layers.Input(shape=(1,), name='movie_input')

    # Strati di embedding
    user_embedding = layers.Embedding(input_dim=num_users, output_dim=embedding_size, name='user_embedding')(user_input)
    movie_embedding = layers.Embedding(input_dim=num_movies, output_dim=embedding_size, name='movie_embedding')(movie_input)

    # Flatten degli embedding
    user_vector = layers.Flatten()(user_embedding)
    movie_vector = layers.Flatten()(movie_embedding)

    # Concatenazione degli embedding utente e film
    concatenated = layers.Concatenate()([user_vector, movie_vector])

    # Strato denso della rete neurale
    dense = layers.Dense(128, activation='relu')(concatenated)
    dense = layers.Dense(64, activation='relu')(dense)
    output = layers.Dense(1)(dense)  # Prevedere un rating singolo

    # Definisci il modello
    model = models.Model(inputs=[user_input, movie_input], outputs=output)

    # Compila il modello
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Mostra il riepilogo del modello
    model.summary()
    
    # Qui puoi inserire il codice per definire e addestrare il modello
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=64,  # Modifica il batch size se necessario
        epochs=10,  # Aumenta il numero di epoche per migliorare l'addestramento
        validation_data=(x_test, y_test)
    )

    # Valutazione del modello sul set di test
    loss = model.evaluate(x_test, y_test)
    print(f"Test Loss: {loss}")
    
    # Esempio di previsione
    user_id = 712664   # Sostituisci con un ID utente valido
    movie_id = 4496  # Sostituisci con un ID film valido

    predicted_rating = model.predict([np.array([user_id]), np.array([movie_id])])
    print(f"Predicted Rating: {predicted_rating}")

    # Grafico della perdita durante l'addestramento
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Perdita del Modello')
    plt.xlabel('Epoche')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./plots/NN/grafico_loss_durante_training_NN_task_1.png')
    return
    


def NN_task_2(df_pca_df, specific_movie_id):
    
    # Shuffle dei dati prima di qualsiasi operazione di split
    df_pca_df = df_pca_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Suddivisione del dataset in due parti basate su 'was_nan'
    train_val_set = df_pca_df[~df_pca_df['was_nan']]  # Righe dove inizialmente non era NaN
    test_set = df_pca_df[df_pca_df['was_nan']]        # Righe dove inizialmente era NaN

    # Verifica e rimuovi duplicati tra train/val set e test set
    test_set_ids = test_set.index
    train_val_set = train_val_set[~train_val_set.index.isin(test_set_ids)]

    # Separare le caratteristiche (X) e l'etichetta (y) per il train/validation set
    X_train_val = train_val_set.drop(columns=['was_nan', specific_movie_id])
    y_train_val = train_val_set[specific_movie_id]

    # Separare le caratteristiche (X) e l'etichetta (y) per il test set
    X_test = test_set.drop(columns=['was_nan', specific_movie_id])
    y_test = test_set[specific_movie_id]

    # Suddivisione del train_val_set in training set e validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42, shuffle=True)

    # Verifica della dimensione dei set
    print(f'Dimensione X_train: {X_train.shape}')
    print(f'Dimensione X_val: {X_val.shape}')
    print(f'Dimensione X_test: {X_test.shape}')
    
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    # Compilazione del modello con Adam optimizer e learning rate ridotto
    model.compile(optimizer=Adam(learning_rate=0.01),
                loss='mean_squared_error',
                metrics=['mean_absolute_error'])

    # Callback per early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size =32,
                    callbacks=[early_stopping],
                    validation_data=(X_val, y_val),
                    shuffle=True)

    # Valutare il modello
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}")
    print(f"Test MAE: {test_mae}")
    
    # Grafico della perdita (loss) durante l'allenamento
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Perdita durante l\'allenamento')
    plt.xlabel('Epoche')
    plt.ylabel('Perdita')
    plt.legend()
    plt.grid(True)
    plt.savefig('./plots/NN/grafico_loss_durante_training_NN_task_2.png')
    
    return