import numpy as np

from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

import torch
import gpytorch
import matplotlib.pyplot as plt

from ..models.train_model import initialize_model, fit_gpytorch_torch

def plot_component_comp(xanes_data, cpts_subset, train_x, test_x, variances, 
                        errors):
    all_pca_model = PCA(n_components=3)
    all_cpts = all_pca_model.fit_transform(xanes_data)

    cth = 0
    train_obj = torch.Tensor(cpts_subset[:, cth]) # grab cth component of training data

    mll, gp_model = initialize_model(train_x, train_obj)
    mll.train()
    mll, info_dict = fit_gpytorch_torch(mll, options={'maxiter':2000, 'lr':0.1})

    gp_model.eval()
    gp_model.likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        obs = gp_model.likelihood(gp_model(test_x), noise=(torch.ones(len(test_x))*0.001))

    plt.close()
    fig, axes = plt.subplots(2,2, figsize=(7,7))

    # variance
    for i, r in enumerate(np.array(variances)):
        axes[0, 0].plot(r, label=f'cpt {i}')
    axes[0, 0].set_xlabel('iter')
    axes[0, 0].set_ylabel('max variance')
    axes[0,0].legend()

    # errors
    for i, r in enumerate(np.array(errors)):
        axes[0, 1].plot(r, label=f'cpt {i}')
    axes[0, 1].set_xlabel('iter')
    axes[0, 1].set_ylabel('Mean squared error')
    axes[0, 1].legend()

    axes[1,0].imshow(all_cpts[:,cth].reshape(32,32), aspect='auto')
    # ax[0,0].imshow(face_image[:,:,1])
    axes[1,0].set_title('ground truth')

    axes[1,1].imshow(obs.mean.reshape(32,32).detach(), aspect='auto')
    axes[1,1].set_title('mean')

    fig.tight_layout()

    return fig, axes

def plot_recon_comp():
    return 1