""" Functions for initializing models, training models"""

import numpy as np

import torch
import gpytorch
from botorch.models.gpytorch import GPyTorchModel
from botorch.optim import optimize_acqf

# initialize GP model
class GridGP(gpytorch.models.ExactGP, GPyTorchModel):
    _num_outputs = 1
    def __init__(self, train_x, train_y, likelihood):
        super(GridGP, self).__init__(train_x, train_y, likelihood)  
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
                                gpytorch.kernels.RBFKernel(ard_num_dims=2) )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def initialize_model(train_x, train_obj):
    noises = torch.ones(len(train_x)) * 0.001
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noises)
    gp_model = GridGP(train_x, train_obj, likelihood)

    gp_model.covar_module.base_kernel.lengthscale = torch.Tensor([1, 1])

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)

    return mll, gp_model

BATCH_SIZE=1
NUM_RESTARTS=1
RAW_SAMPLES=32

def optimize_acq_and_observe(acq_func, bounds=torch.Tensor([[0., 31.], [0., 31.]]).T):
    """optimize_acq_and_observe [summary]

    [extended_summary]

    :param acq_func: [description]
    :type acq_func: [type]
    :param bounds: [description], defaults to torch.Tensor([[0., 31.], [0., 31.]]).T
    :type bounds: [type], optional
    :return: [description]
    :rtype: [type]
    """
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )

    # observe new values
    new_x = candidates.detach()

    return new_x

from botorch.optim.fit import fit_gpytorch_torch
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from .acquisition import MaxVariance

def BO_loop(xanes_data, 
            BO_iters=100, n_init=10, n_cpts=3, seed=10, acq_func=MaxVariance,
             optim_options={'maxiter':2000, 'lr':0.1, 'disp':False}):

    """
    Run Bayesian Optimization loop in latent space.  
    
    Uses PCA latent space as objective to optimize.  At each loop, fits a GP for each 
    component (n_cpts).  
    - Selects next point to 'measure' by argmax(explained_variance_ratio * acq_func_value)
    - Returns error and variance at each iteration (for each component where applicable)
    - error calculated as MSE(measured, PCA_reconstructed), [n-iters, 1]
    - variance returned is max variance for each component's GP [n_cpts, n_iters]

    :return: [description]
    :rtype: [type]
    """
    rng = np.random.default_rng(seed=seed)

    init_inds = rng.choice(1024, size=(n_init), replace=False)
    init_locs = np.array([[int(i/32), i%32] for i in init_inds])
    xanes_subset = xanes_data[init_inds]

    all_pca_model = PCA(n_components=n_cpts)
    all_cpts = all_pca_model.fit_transform(xanes_data)
    
    pca_model = PCA(n_components=n_cpts)
    cpts_subset = pca_model.fit_transform(xanes_subset)

    # initialization: vector valued GP
    train_x = torch.Tensor(init_locs)
    train_obj = torch.Tensor(cpts_subset[:, 0])
    x, y = torch.meshgrid(torch.linspace(0,31,32), torch.linspace(0,31,32))
    test_x = torch.vstack((x.flatten(), y.flatten())).T

    # initial view
    mll, gp_model = initialize_model(train_x, train_obj)
    mll.train()
    fit_gpytorch_torch(mll, options=optim_options)

    gp_model.eval()
    gp_model.likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        init_obs = gp_model.likelihood(gp_model(test_x), noise=(torch.ones(len(test_x))*0.001))

    # Grab new subset of data 
    xanes_subset = xanes_data[[int(torch.round(32*i+j)) for i, j in train_x]]

    # redo PCA and ground_truth based on current training set
    pca_model = PCA(n_components=n_cpts)
    cpts_subset = pca_model.fit_transform(xanes_subset)

    variances = [[] for i in range(n_cpts)]
    cand_pts  = [[] for i in range(n_cpts)]
    errors = []

    chosen = []
    # BO Loops
    for i in range(BO_iters):

        print(f'=== iter {i}, ({cpts_subset.shape})')
        gps = [initialize_model(train_x, torch.Tensor(cpts_subset[:, c])) for c in range(n_cpts)]
        means = [torch.ones(test_x.shape) for x in range(n_cpts)]
        
        for c in range(n_cpts): # grab candidate for each component
            print(f'-- cpt {c}')

            mll, gp_model = gps[c]
            mll.train()
            fit_gpytorch_torch(mll, options=optim_options)

            acq = acq_func(gp_model)
            cands = optimize_acq_and_observe(acq)
        
            gp_model.eval()
            gp_model.likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                mid_obs = gp_model.likelihood(gp_model(test_x), 
                                        noise=(torch.ones(len(test_x))*0.001))
        
            variances[c].append(mid_obs.variance.max().item())
            cand_pts[c].append(cands[0].int().detach())

            means[c] = mid_obs.mean.detach().numpy() # [1024]

        # generate reconstructions via pca_model
        recon_data = pca_model.inverse_transform(np.array(means).T)


        # PCA is unstable w.r.t. sign of components, reconstructions might be inverted. 
        best_err = np.min( 
                    [mean_squared_error( all_cpts[:,c], -mid_obs.mean.detach().flatten()), 
                    mean_squared_error( all_cpts[:,c], mid_obs.mean.detach().flatten())]
                    )
        
        errors.append( mean_squared_error(recon_data, xanes_data) )
        # Select new point
          
        # weight max variances by explained variance ratio, use as criterion
        selected_cpt = np.argmax(pca_model.explained_variance_ratio_ * np.array(variances)[:,i])
        chosen.append(selected_cpt)
        train_x_new = torch.vstack([ train_x, cand_pts[selected_cpt][i] ])
        train_x = train_x_new

        # Grab new subset of data 
        xanes_subset = xanes_data[[int(32*i.item())+int(j.item()) for i, j in train_x]]

        # redo PCA and ground_truth based on current training set
        pca_model = PCA(n_components=n_cpts)
        cpts_subset = pca_model.fit_transform(xanes_subset)

    return gp_model, init_obs, train_x, test_x, variances, errors, cpts_subset, chosen