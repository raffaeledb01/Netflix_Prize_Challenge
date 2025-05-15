# 🎬 Netflix Prize Challenge
This repository contains the implementation of various statistical and machine learning models aimed at predicting user ratings for movies, as part of the Netflix Prize Challenge. The objective is to minimize the prediction error (RMSE) by leveraging different dimensionality reduction techniques, neural networks, and hybrid approaches.

---

## 📚 Project Description
The Netflix Prize was a competition held to improve the accuracy of movie recommendation systems. This project replicates a similar environment using a dataset of user ratings and explores multiple approaches to predict unseen ratings. The models are evaluated primarily using the Root Mean Squared Error (RMSE).

---

## 🧠 Implemented Methods
This project explores and compares several recommendation strategies:

### Singular Value Decomposition (SVD)
A matrix factorization technique that decomposes the user-item rating matrix into latent features, enabling collaborative filtering based on learned patterns.

### Non-negative Matrix Factorization (NMF)
A matrix factorization method constrained to non-negative values. It helps interpret the latent features as additive components. Grid search was used for hyperparameter tuning.

### Autoencoders
Neural networks trained to reconstruct the input rating matrix. These models capture complex, non-linear relationships between users and items.

### Feedforward Neural Networks (NN)
A basic multilayer perceptron trained to predict user ratings, using user and movie indices as inputs encoded with embeddings.

### Principal Component Analysis (PCA)
Used to reduce the dimensionality of the rating matrix, retaining the most significant variance components.

### Preprocessing Techniques
Custom tools for filtering, normalizing, splitting, and aggregating data—key to preparing robust training datasets and analyzing rating distributions.

---

## 📁 Repository Structure

```

/Netflix_Prize_Challange
├──📁 code/                             # Python source files
│   ├── autoencoder.py                # Autoencoder model and training logic
│   ├── main.py                       # Entry point to run experiments and compare models
│   ├── NMF.py                        # Implementation of Non-negative Matrix Factorization
│   ├── NN.py                         # Feedforward Neural Network recommender
│   ├── PCA.py                        # PCA for dimensionality reduction
│   ├── preprocessing.py              # Data cleaning, splitting, and formatting functions
│   └── SVD.py                        # Singular Value Decomposition for matrix factorization
│
├──📁 dataset/                          # Directory for the Netflix dataset (not included due to size)
│
├──📁 plots/                            # All result visualizations
│   ├──📁 autoencoder/                  # Autoencoder training curve
│   │   └── learning_curve.png
│   ├──📁 nmf/                          # NMF performance metrics and grid search results
│   │   ├── nmf_grid_search_results_rmse_heatmap.png
│   │   ├── rmse_vs_epochs.jpg
│   │   └── training_validation_loss_50_epochs.png
│   ├──📁 nn/                           # NN training diagnostics
│   │   └── grafico_1.png
│   ├──📁 preproc/                      # Preprocessing insights and rating distributions
│   │   ├── distribuzione_rating.png
│   │   ├── static_movie_ratings_by_day.png
│   │   ├── static_movies_grouped_by_year.png
│   │   ├── static_ratings_per_movie_NEW.png
│   │   └── static_ratings_per_user_NEW.png
│   └──📁 svd/                          # SVD model performance visualization
│       ├── heatmap_rmse.png
│       ├── rmse_vs_epochs.png
│       └── training_validation_loss_NEW.png
│
├── Netflix_Prize.pdf                 # Final report describing methodology, results, and evaluation
├── README.md                         # Project Documentation
└── LICENSE                           # MIT License
```

---

## 📊 Results
Autoencoder and NN models generalize well when trained with proper regularization and epochs.

SVD and NMF offer strong baseline performance with low computational cost.

Visual results in /plots/ highlight model comparisons, error trends, and data insights.

---

## ⚙️ Requirements
Before running the code, make sure to install the required Python packages. It is recommended to use a virtual environment.

---

## 🐍 Python Version
Python 3.8 or higher

---

## 📦 Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

---

## 📑 Report
A detailed summary of the project, including theoretical background, methodology, and result interpretation, is available in:

- 📄 Netflix_Prize.pdf

---

## 👨‍💻 Authors

- Raffaele Di Benedetto
- Antonio Nardone

---

## 📜 License
This project is released under the MIT License.