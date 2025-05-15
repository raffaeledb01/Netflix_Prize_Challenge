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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from scipy import sparse
from scipy.linalg import svd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
import os

def autoencoder(df, user_movie_matrix):
    
    # Step 2: Definizione dell'Autoencoder
    num_users, num_movies = user_movie_matrix.shape
    encoding_dim = 64  # Dimensionalità dello spazio latente
    
    input_layer = Input(shape=(num_movies,))
    encoder = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(input_layer)
    encoder = Dropout(0.2)(encoder)
    encoder = Dense(256, activation='relu')(encoder)
    encoder = Dropout(0.2)(encoder)
    encoder = Dense(encoding_dim, activation='relu')(encoder)

    decoder = Dense(256, activation='relu')(encoder)
    decoder = Dense(512, activation='relu')(decoder)
    decoder = Dense(num_movies, activation='linear')(decoder)
    
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    train_data, test_data = train_test_split(user_movie_matrix, test_size=0.2, random_state=42)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = autoencoder.fit(train_data, train_data,
                            epochs=100,
                            batch_size=32,
                            validation_data=(test_data, test_data),
                            verbose=2,
                            callbacks=[early_stopping, reduce_lr])
    
    # recupero parametri per creare grafico
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(train_loss) + 1)
    
    # Creazione del grafico
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('autoencoder.png')
    
    
    return