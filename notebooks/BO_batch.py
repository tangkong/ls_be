#!/usr/bin/env python
# coding: utf-8

import sys

import numpy as np
import pickle
import time
import datetime

from pathlib import Path
from src.data.make_dataset import generate_dataset
from src.models.train_model import BO_loop, grid_search, dist_loop
from src.models.acquisition import Random, MaxVariance

from functools import partial


def main(basis='xanes', seed=42, iters=5,
        save_path= 'C:\\Users\\roberttk\\Desktop\\SLAC_RA\\machine-learning\\ls_be\\notebooks'):# '/sdf/home/r/roberttk/results_trig_400/'):
    
    seed = int(seed)
    iters = int(iters)

    rng = np.random.default_rng(seed=seed)
    data_seeds = rng.integers(low=0, high=10000, size=iters)
    print(data_seeds)
    ds_cfg = {'n_cpts': 5, 'supply_truth':False, 'basis':basis}
    bo_cfg = {'pca_cpts': 4, 'bo_iters':4}

    mv_bo_loop = partial(BO_loop, acq_func=MaxVariance )
    mv_bo_loop.__name__ = 'mv_bo_loop'
    rand_bo_loop = partial(BO_loop, acq_func=Random )
    rand_bo_loop.__name__ = 'rand_bo_loop'

    loops = [grid_search, dist_loop, mv_bo_loop, rand_bo_loop]
    results_path = Path(save_path)
    for s in data_seeds:
        _, data, _ = generate_dataset(n_cpts=ds_cfg['n_cpts'], 
                                    seed=s, supply_truth=ds_cfg['supply_truth'],
                                    xanes=ds_cfg['basis'])
        for loop in loops:
            # run test
            ts = time.time()
            print(f'{loop.__name__}: seed {s} @ time {ts}')
            print(datetime.datetime.fromtimestamp(ts).isoformat())
            _, varis, errs, info_dict = loop(data, n_cpts=bo_cfg['pca_cpts'], 
                                            n_iters=bo_cfg['bo_iters'])

            # construct results dictionary and save
            results = {'max_variances': varis,
                        'errors': errs,
                        'info_dict': info_dict,
                        'bo_cfg': bo_cfg,
                        'ds_cfg': ds_cfg,
                        'loop_type': loop.__name__,
                        'start_time': ts
                        }

            with open(results_path / f'results_{ds_cfg["basis"]}_{s}_{loop.__name__}.pkl', 'wb') as f:
                pickle.dump(results, f)    

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print('no arguments supplied, running with default parameters')
        main()
    elif sys.argv[1] =='help':
        print('Assumed argument order: basis, seed, iters, save_path')
    else:
        main(basis=sys.argv[1], seed=sys.argv[2], 
                iters=sys.argv[3], save_path=sys.argv[4])

    #main()