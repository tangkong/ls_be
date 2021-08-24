""" Functions for initializing models, training models"""

import numpy as np

import torch
import gpytorch
from botorch.models.gpytorch import GPyTorchModel
from botorch.optim import optimize_acqf

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

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
RAW_SAMPLES=int(32*32/4)

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

def ret_angle(loc1, loc2):
    loc = loc1 - loc2
    if np.linalg.norm(loc) == 0:
        return 0
    return np.arccos( np.dot(loc, np.array([0,1])) / np.linalg.norm(loc) )

def setup_figure():
    fig, ax = plt.subplots(2,1, figsize=(5, 10))
    ax[0].set_xlabel('iters')
    ax[0].set_ylabel('spectra reconstruction error')
    return fig, ax

def BO_loop(xanes_data, 
            n_iters=100, n_init=10, n_cpts=3, seed=10, acq_func=MaxVariance,
            show_iter_output=True,
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
    errors = {'pca_mse':[], 'spec_mse_avg':[], 'spec_mse_std':[]}

    chosen = []

    # fig, axes = setup_figure()
    # ims = []
    # BO Loops
    for i in range(n_iters):

        if show_iter_output:
            print(f'=== iter {i}, ({cpts_subset.shape})', end=' ')

        gps = [initialize_model(train_x, torch.Tensor(cpts_subset[:, c])) for c in range(n_cpts)]
        means = [torch.ones(test_x.shape) for x in range(n_cpts)]
        
        for c in range(n_cpts): # grab candidate for each component
            if show_iter_output:
                print(f'-- cpt {c}', end=' ')

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

        if show_iter_output:
            print(' ')

        # collect error information
        ## generate reconstructions via pca_model
        recon_data = pca_model.inverse_transform(np.array(means).T)
        spectra_mses = [mean_squared_error(recon_data[i,:], xanes_data[i,:]) 
                                    for i in range(1024) ]
        errors['spec_mse_avg'].append( np.mean(spectra_mses) )
        errors['spec_mse_std'].append(np.std( spectra_mses ))

        pca_err = np.min( 
            [mean_squared_error( all_cpts[:,0], -means[0].flatten()), 
            mean_squared_error( all_cpts[:,0], means[0].flatten())]
            )

        errors['pca_mse'].append(pca_err)

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

    #     if i % 10 == 0:
    #         # generate animation
    #         im_pca_err = axes[0].plot(errors['spec_mse_avg'], c='b')
    #         im_map = axes[1].imshow(means[0].reshape(32,32))
    #         im_scatter = axes[1].scatter(train_x[:, 1], train_x[:, 0],
    #                         s=100, marker='s', c='r')
    #         ims.append([im_pca_err[0], im_map, im_scatter])

    # writer = PillowWriter(fps=0.5)
    # ani = animation.ArtistAnimation(fig, ims, interval=10)
    # ani.save('fp_BO.gif', writer=writer)


    info_dict = { 'init_obs': init_obs, 
                    'train_x':train_x, 
                    'test_x': test_x,
                    'chosen': chosen,
                    'curr_cpt_weights': cpts_subset
                    }

    return gps, variances, errors, info_dict

def grid_search(xanes_data, n_iters=100, n_cpts=3, show_iter_output=True, 
                  optim_options={'maxiter':2000, 'lr':0.1, 'disp':False}):
    """ Control test """

    init_inds = np.array(range(n_cpts)) # need at least n_cpts datapts to do PCA
    init_locs = np.unravel_index(init_inds, (32,32))
    xanes_subset = xanes_data[init_inds]

    all_pca_model = PCA(n_components=n_cpts)
    all_cpts = all_pca_model.fit_transform(xanes_data)
        
    pca_model = PCA(n_components=n_cpts)
    cpts_subset = pca_model.fit_transform(xanes_subset)

    # initialization
    train_x = torch.Tensor(init_locs).T
    train_obj = torch.Tensor(cpts_subset[:,0]) # train on first component

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

    variances = [[] for i in range(n_cpts)]
    cand_pts  = [[] for i in range(n_cpts)]
    errors = {'pca_mse':[], 'spec_mse_avg':[], 'spec_mse_std':[]}

    # fig, axes = setup_figure()
    # ims = []
    for i in range(n_iters):
        if show_iter_output:
            print(f'=== iter {i}, ({cpts_subset.shape})', end=' ')

        gps = [initialize_model(train_x, torch.Tensor(cpts_subset[:, c])) for c in range(n_cpts)]
        means = [torch.ones(test_x.shape) for x in range(n_cpts)]
        for c in range(n_cpts): # grab candidate for each component
            if show_iter_output:
                print(f'-- cpt {c}', end=' ')

            mll, gp_model = gps[c]
            mll.train()
            fit_gpytorch_torch(mll, options=optim_options)

            gp_model.eval()
            gp_model.likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                mid_obs = gp_model.likelihood(gp_model(test_x), 
                                        noise=(torch.ones(len(test_x))*0.001))
        
            variances[c].append(mid_obs.variance.max().item())
            cand_pts[c].append(n_cpts+1)
            means[c] = mid_obs.mean.detach().numpy() # [1024]

        if show_iter_output:
            print(' ')

        # collect error information
        ## generate reconstructions via pca_model
        recon_data = pca_model.inverse_transform(np.array(means).T)
        spectra_mses = [mean_squared_error(recon_data[i,:], xanes_data[i,:]) 
                                    for i in range(1024) ]
        errors['spec_mse_avg'].append( np.mean(spectra_mses) )
        errors['spec_mse_std'].append(np.std( spectra_mses ))

        pca_err = np.min( 
            [mean_squared_error( all_cpts[:,0], -means[0].flatten()), 
            mean_squared_error( all_cpts[:,0], means[0].flatten())]
            )

        errors['pca_mse'].append(pca_err)

        # select new points
        train_inds = np.array(range(n_cpts + i + 1))
        train_x = torch.Tensor(np.unravel_index(train_inds, (32,32))).T

        # Grab new subset of data 
        xanes_subset = xanes_data[np.array(range(n_cpts+i+1))]

        # redo PCA and ground_truth based on current training set
        pca_model = PCA(n_components=n_cpts)
        cpts_subset = pca_model.fit_transform(xanes_subset)

    #     if i % 10 == 0:
    #         # generate animation
    #         im_pca_err = axes[0].plot(errors['spec_mse_avg'], c='b')
    #         im_map = axes[1].imshow(means[0].reshape(32,32))
    #         im_scatter = axes[1].scatter(train_x[:, 1], train_x[:, 0],
    #                         s=100, marker='s', c='r')
    #         ims.append([im_pca_err[0], im_map, im_scatter])

    # writer = PillowWriter(fps=0.5)
    # ani = animation.ArtistAnimation(fig, ims, interval=10)
    # ani.save('fp_grid.gif', writer=writer)

    # snapshot final info
    info_dict = { 'init_obs': init_obs, 
                        'train_x':train_x, 
                        'test_x': test_x,
                        'curr_cpt_weights': cpts_subset
                        }

    return gps, variances, errors, info_dict

from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors

def dist_loop(xanes_data, n_iters=100, n_init=10, n_cpts=3, seed=10,
                dist_metric=cosine_distances, show_iter_output=True, 
                optim_options={'maxiter':2000, 'lr':0.1, 'disp':False}):
    """ Try heuristic method based on nearest neighbor distances
    Stores data in a symmetric matrix
    
    Currently assumes image is 32 x 32
    xanes_data: (n_samples, len(spectrum)) = (1024, len)
    _loc = real space
    _ind = raveled, singular index
    """
    # initialize storage
    dist_matrix = np.zeros((xanes_data.shape[0], xanes_data.shape[0]))

    # initialize training points
    rng = np.random.default_rng(seed=seed)
    init_inds = rng.choice(xanes_data.shape[0], size=(n_init), replace=False)

    all_pca_model = PCA(n_components=n_cpts)
    all_cpts = all_pca_model.fit_transform(xanes_data)

    pca_model = PCA(n_components=n_cpts)
    cpts_subset = pca_model.fit_transform(xanes_data[init_inds])

    for i in init_inds:
        for j in init_inds:
            distance = dist_metric(xanes_data[i].reshape(1,-1), 
                                    xanes_data[j].reshape(1,-1)).item()
            dist_matrix[i, j] = distance
            dist_matrix[j, i] = distance

    # Look for nearest neighbors in real space... 
    x, y = torch.meshgrid(torch.linspace(0,31,32), torch.linspace(0,31,32))
    test_x = torch.vstack((x.flatten(), y.flatten())).T

    tested_locs = np.unravel_index(init_inds, (32,32))
    tested_locs = np.array([tested_locs[0], tested_locs[1]]).T
    init_obs = torch.Tensor(tested_locs.copy())

    variances = [[] for i in range(n_cpts)]
    cand_pts  = []
    errors = {'pca_mse':[], 'spec_mse_avg':[], 'spec_mse_std':[]}

    # fig, axes = setup_figure()
    # ims = []
    for i in range(n_iters):
        if show_iter_output:
            print(f'=== iter {i},', end=' ')
        # each iteration of optimization, updating tested_locs
        neigh = NearestNeighbors(n_neighbors=3, radius=1024)
        neigh.fit(tested_locs)

        # init mask
        mask = np.zeros_like(dist_matrix) 
        # for each measured point, find nearest neighbors from measured points
        for loc in tested_locs:
            # from initialized 0 mask, set mask to 1 with row=from, col=to
            neigh_dist, neigh_inds = neigh.kneighbors([loc], 4)
            neigh_dist = neigh_dist[0]
            neigh_inds = neigh_inds[0]
            # sort out colinear nns
            nn_angs = np.nan_to_num([ret_angle(x, loc) for x in tested_locs[neigh_inds]], 0)
            to_remove = []
            for ang in nn_angs:
                if np.isnan(ang): # skips the self-comparison
                    continue
                temp_angs = np.abs(nn_angs - ang)
                thresh = 0.1
                if np.sum(temp_angs < thresh) > 1: # if there are more than one at this similar angle
                    # find max distance between similar angles
                    max_dist_ind = np.argmax( (temp_angs<thresh) * neigh_dist )
                    # remove further item from list
                    to_remove.append(max_dist_ind)
                    
            # delete items from nn list
            neigh_inds = np.delete(neigh_inds, to_remove)
            
            nn_inds = np.ravel_multi_index(tested_locs[neigh_inds].T, (32,32))
            loc_ind = np.ravel_multi_index(loc, (32,32))
            
            mask[loc_ind, nn_inds] = 1
            mask[loc_ind, loc_ind] = 0

        # mask distance matrix
        masked = mask * dist_matrix

        # find max distance, next point is between
        mins = np.where(masked==masked.max())
        from_loc = np.unravel_index(mins[0][0], (32,32))
        from_loc = np.array(from_loc).flatten()
        to_loc = np.unravel_index(mins[1][0], (32,32))
        to_loc = np.array(to_loc).flatten()

        next_loc = np.round(np.average([from_loc, to_loc], axis=0)).astype(int)
        its = 0
        shift_x = True
        while any([all(next_loc==x) for x in tested_locs]) and (its < 10):
            # if we're trying to re-measure a point, shift it?
            if shift_x:
                next_loc += np.array([1, 0]) 
                shift_x = not shift_x
            else: 
                next_loc += np.array([0, 1])
                shift_x = not shift_x
            
            next_loc = next_loc % 32
            its += 1
        print(f'shifted {its}x')
        # if still colliding, pick random point...
        if any([all(next_loc==x) for x in tested_locs]):
            tested_inds = tested_inds = np.ravel_multi_index(tested_locs.T, (32,32))
            test_x_inds = np.ravel_multi_index(test_x.T.numpy().astype(int), (32,32))
            untested_inds = np.setdiff1d(test_x_inds, tested_inds)
            new_ind = rng.choice(untested_inds)

            next_loc = np.array(np.unravel_index(new_ind, (32,32)))
            print('picked random point')

        print(f'({from_loc}-->{to_loc} = {next_loc})', end=' ')

        # collect error information
        ## generate reconstructions via pca_model
        gps = [initialize_model(torch.Tensor(tested_locs), torch.Tensor(cpts_subset[:, c])) for c in range(n_cpts)]
        means = [torch.ones(test_x.shape) for x in range(n_cpts)]
        for c in range(n_cpts): # grab candidate for each component
            if show_iter_output:
                print(f'-- cpt {c}', end=' ')

            mll, gp_model = gps[c]
            mll.train()
            fit_gpytorch_torch(mll, options=optim_options)

            gp_model.eval()
            gp_model.likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                mid_obs = gp_model.likelihood(gp_model(test_x), 
                                        noise=(torch.ones(len(test_x))*0.001))
        
            variances[c].append(mid_obs.variance.max().item())
            cand_pts.append(next_loc)
            means[c] = mid_obs.mean.detach().numpy() # [1024]

        if show_iter_output:
            print(' ')

        recon_data = pca_model.inverse_transform(np.array(means).T)
        spectra_mses = [mean_squared_error(recon_data[i,:], xanes_data[i,:]) 
                                    for i in range(1024) ]
        errors['spec_mse_avg'].append( np.mean(spectra_mses) )
        errors['spec_mse_std'].append(np.std( spectra_mses ))

        pca_err = np.min( 
            [mean_squared_error( all_cpts[:,0], -means[0].flatten()), 
            mean_squared_error( all_cpts[:,0], means[0].flatten())]
            )

        errors['pca_mse'].append(pca_err)

        # add new point to list
        tested_locs = np.concatenate((tested_locs, next_loc.reshape(1,-1)), axis=0).astype(int)
        tested_inds = np.ravel_multi_index(tested_locs.T, (32,32))
        # update distance matrix
        for j in tested_inds:
            i = tested_inds[-1]
            distance = dist_metric(xanes_data[i].reshape(1, -1), 
                                    xanes_data[j].reshape(1, -1)).item()
            dist_matrix[i, j] = distance
            dist_matrix[j, i] = distance


        # retrain PCA model
        pca_model = PCA(n_components=n_cpts)
        cpts_subset = pca_model.fit_transform(xanes_data[tested_inds])

    #     if i % 10 == 0:
    #         # generate animation
    #         im_pca_err = axes[0].plot(errors['spec_mse_avg'], c='b')
    #         im_map = axes[1].imshow(means[0].reshape(32,32))
    #         im_scatter = axes[1].scatter(tested_locs[:, 1], tested_locs[:, 0],
    #                         s=100, marker='s', c='r')
    #         ims.append([im_pca_err[0], im_map, im_scatter])

    # writer = PillowWriter(fps=0.5)
    # ani = animation.ArtistAnimation(fig, ims, interval=10)
    # ani.save('fp_dist.gif', writer=writer)
    
        

    # snapshot final info
    info_dict = { 'init_obs': init_obs, 
                        'train_x':tested_locs, 
                        'test_x': test_x,
                        'curr_cpt_weights': cpts_subset
                        }

    return gps, variances, errors, info_dict