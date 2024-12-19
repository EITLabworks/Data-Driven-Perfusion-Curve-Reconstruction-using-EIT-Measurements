import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .util import Pearson_correlation
import numpy as np


def plot_pca(data):
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data_standardized)

    plt.figure(figsize=(10, 7))
    plt.scatter(pca_data[:, 0], pca_data[:, 1], s=1, alpha=0.5)
    plt.title("2-Component PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid()
    plt.show()


def plot_relative_error_aorta(Y_true, Y_pred, std, var, mean, s_name=None):
    rel_mae_err = np.mean(np.abs((Y_pred - Y_true) / Y_true), axis=0) * 100
    std_err = np.std((Y_pred - Y_true) / Y_true, axis=0) * 100  # np.abs()?
    var_err = np.var((Y_pred - Y_true) / Y_true, axis=0) * 100  # np.abs()?

    L = Y_true.shape[0]

    MAE = np.sum(np.abs(Y_true - Y_pred)) / 1024 / L
    MSE = np.sum((Y_true - Y_pred) ** 2) / 1024 / L
    PN = np.mean(Pearson_correlation(Y_true, Y_pred))

    print("MAE", MAE)
    print("MSE", MSE)
    print("Pearson number", PN)

    plt.figure(figsize=(6, 2))
    if mean:
        plt.plot(rel_mae_err, label="Relative mean absolute error")
    if std:
        plt.plot(std_err, label="Standard deviation")
    if var:
        plt.plot(var_err, label="Variance")
    plt.legend()
    plt.xlabel("Aorta pressure curve index")
    plt.xticks(ticks=np.linspace(0, 1024, 5), labels=np.linspace(0, 1024, 5, dtype=int))
    plt.ylabel("Relative error (%)")
    plt.grid()
    plt.tight_layout()
    if s_name != None:
        plt.savefig(s_name)
    plt.show()

    return MAE, MSE, PN


def plot_random_predictions(Y_true, Y_pred, n=4):
    r_idx = np.random.randint(0, Y_true.shape[0], size=n)
    for idx in r_idx:
        print(f"Selected prediction {idx}.")
        plt.figure(figsize=(6, 3))
        plt.plot(Y_pred[idx], label="Pred")
        plt.plot(Y_true[idx], label="True")
        plt.legend()
        plt.grid()
        plt.show()
