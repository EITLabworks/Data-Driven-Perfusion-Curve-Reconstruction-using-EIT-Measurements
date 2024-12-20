import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def output_err_for_LaTeX(Y_true, Y_pred, f_name):
    rel_mae_err = np.mean(np.abs((Y_pred - Y_true) / Y_true), axis=0) * 100
    std_err = np.std(np.abs((Y_pred - Y_true) / Y_true), axis=0) * 100  # np.abs()?
    var_err = np.var(np.abs((Y_pred - Y_true) / Y_true), axis=0) * 100  # np.abs()?
    latex_export = {
        "x": np.arange(1, 1025),
        "rel_mae_err": rel_mae_err,
        "std_err": std_err,
        "var_err": var_err,
    }
    pd.DataFrame(latex_export).to_csv(f_name, index=False)


def output_curve_for_LaTeX(Y_true, Y_pred, f_name, n_samples=4, plot=True):
    assert Y_pred.shape[0] == Y_true.shape[0]
    selcts = np.random.randint(0, Y_pred.shape[0], size=n_samples)
    latex_export = {
        "x": np.arange(1, 1025),
    }
    for i, idx in enumerate(selcts):
        latex_export[f"y_true_{i}"] = Y_true[idx]
        latex_export[f"y_pred_{i}"] = Y_pred[idx]
        if plot:
            plt.title(f"Plot {i} of sample {idx}")
            plt.plot(Y_true[idx], label="true")
            plt.plot(Y_pred[idx], label="pred")
            plt.legend()
            plt.show()
    pd.DataFrame(latex_export).to_csv(f_name, index=False)
