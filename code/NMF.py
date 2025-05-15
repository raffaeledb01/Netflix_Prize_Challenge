import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Reader, Dataset, NMF
from surprise.model_selection import GridSearchCV


def NMF(df):
    
    reader = Reader()

    print('Creazione dataset per NMF')
    # get just top 100K rows for faster run time
    data = Dataset.load_from_df(df[['Cust_Id', 'Movie_Id', 'Rating']][:1000000], reader)

    # Configura i parametri per GridSearchCV con il modello NMF
    param_grid_svd = {
        'n_factors': [13,20.25], # inizialmente 80,100,120
        'n_epochs': [15,20,25], # inizialmente 10,20,30
        'reg_qi': [0.01,0.05],
        'reg_pu': [0.01,0.05]
    }

    gs_svd = GridSearchCV(NMF, param_grid_svd, measures=['rmse', 'mae'], cv=5)
    gs_svd.fit(data)

    # Mostra i migliori parametri e il corrispondente risultato
    print("Best RMSE score for NMF:", gs_svd.best_score['rmse'])
    print("Best parameters for NMF (RMSE):", gs_svd.best_params['rmse'])
    print("Best MAE score for NMF:", gs_svd.best_score['mae'])
    print("Best parameters for NMF (MAE):", gs_svd.best_params['mae'])


    # Supponiamo che tu abbia già eseguito la grid search e hai il DataFrame con i risultati
    results_df = pd.DataFrame.from_dict(gs_svd.cv_results)
    print(results_df)

    results_df.to_csv('results_NMF.csv', index=False)



    print("Generazione del grafico dei Valori Singolari...")
    try:
        U, sigma, Vt = np.linalg.svd(df_p.fillna(0), full_matrices=False)
    except Exception as e:
        print(f"Errore nella creazione del grafico dei valori singolari: {e}")

    print("Addestramento del modello e calcolo del RMSE durante l'addestramento...")
    try:
        epochs = list(range(1, gs_svd.best_params['rmse']['n_epochs'] + 1))

        train_rmse = []
        algo = NMF(n_factors=gs_svd.best_params['rmse']['n_factors'], 
                n_epochs=gs_svd.best_params['rmse']['n_epochs'],
                lr_all=gs_svd.best_params['rmse']['lr_all'], 
                reg_all=gs_svd.best_params['rmse']['reg_all'])
        for epoch in epochs:
            algo.n_epochs = epoch
            algo.fit(data.build_full_trainset())
            predictions = algo.test(data.build_full_trainset().build_testset())
            rmse = np.sqrt(np.mean([(pred.r_ui - pred.est)**2 for pred in predictions]))
            train_rmse.append(rmse)
    except Exception as e:
        print(f"Errore durante l'addestramento del modello e il calcolo del RMSE: {e}")

    print("Creazione della pivot table...")
    try:
        pivot_table = results_df.pivot_table(index='param_n_epochs', columns='param_n_factors', values='mean_test_rmse', aggfunc='mean')
    except Exception as e:
        print(f"Errore durante la creazione della pivot table: {e}")

    print("Recupero del miglior modello per RMSE e addestramento sul set di dati completo...")
    try:
        best_svd = gs_svd.best_estimator['rmse']
        trainset = data.build_full_trainset()
        best_svd.fit(trainset)
        testset = trainset.build_testset()
        predictions = best_svd.test(testset)
        residuals = [pred.r_ui - pred.est for pred in predictions]
    except Exception as e:
        print(f"Errore durante l'addestramento e la previsione del miglior modello NMF: {e}")


    print("Creazione della Learning Curve RMSE...")
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_rmse, marker='o')
        plt.title('Learning Curve: RMSE durante l\'addestramento')
        plt.xlabel('Numero di Epoche')
        plt.ylabel('RMSE')
        plt.savefig('plots/learning_curve_rmse.png')
        plt.close()
    except Exception as e:
        print(f"Errore durante la creazione della Learning Curve RMSE: {e}")

    print("Creazione della Heatmap RMSE...")
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlGnBu")
        plt.title('Heatmap RMSE in funzione di n_factors e n_epochs')
        plt.xlabel('n_factors')
        plt.ylabel('n_epochs')
        plt.savefig('plots/heatmap_rmse.png')
        plt.close()
    except Exception as e:
        print(f"Errore durante la creazione della heatmap RMSE: {e}")
        

        # Parametri specificati
    n_factors = 89
    n_epochs = 50
    lr_all = 0.005
    reg_all = 0.1

    # Dividi il dataset in training e test set
    trainset, testset = train_test_split(data, test_size=0.2)

    # Liste per memorizzare le perdite
    train_losses = []
    val_losses = []

    # Esegui il ciclo di addestramento per ogni epoca
    for epoch in range(1, n_epochs + 1):
        print('Sto runnando epoca:' + str(epoch))
        algo = SVD(n_factors=n_factors, n_epochs=epoch, lr_all=lr_all, reg_all=reg_all)
        algo.fit(trainset)

        # Predizioni sul training set
        train_predictions = algo.test(trainset.build_testset())
        train_rmse = accuracy.rmse(train_predictions, verbose=False)
        train_losses.append(train_rmse)

        # Predizioni sul validation set
        test_predictions = algo.test(testset)
        test_rmse = accuracy.rmse(test_predictions, verbose=False)
        val_losses.append(test_rmse)

    # Crea il grafico
    epochs = list(range(1, n_epochs + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss (RMSE)', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss (RMSE)', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('./plots_SVD/training_validation_loss_50_epochs.png')
    plt.close()
    
    
    
    def rmse_vs_factors(algorithm, data):
    
        rmse_algorithm = []
        
        for k in range(79, 91, 1):
            algo = algorithm(n_factors = k, n_epochs= 100, lr_all=0.005, reg_all=0.1)
            print('Sto runnando con epoche: ' + str(k))

            
            #["test_rmse"] is a numpy array with min accuracy value for each testset
            loss_fce = cross_validate(algo, data, measures=['RMSE'], cv=3, verbose=True)["test_rmse"].mean() 
            rmse_algorithm.append(loss_fce)
        
        return rmse_algorithm
        


    def plot_rmse(rmse, algorithm):
    
        print('sto generando il grafico')
        plt.figure(num=None, figsize=(11, 5), dpi=80, facecolor='w', edgecolor='k')

        plt.subplot(2,1,1)
        plt.plot(rmse)
        plt.xlim(0,100)
        plt.title("{0} Performance: RMSE Against Number of Factors".format(algorithm), size = 20 )
        plt.ylabel("Mean RMSE (cv=5)")

        plt.subplot(2,1,2)
        plt.plot(rmse)
        plt.xlim(0,50)
        plt.xticks(np.arange(0, 52, step=2))

        plt.xlabel("{0}(n_factor = k)".format(algorithm))
        plt.ylabel("Mean RMSE (cv=5)")
        plt.axvline(np.argmin(rmse), color = "r")
        plt.savefig('./plots_NMF/rmse_vs_epochs.png')
        plt.close()

    rmse_svd = rmse_vs_factors(NMF,data)

    plot_rmse(rmse_svd,"NMF")
